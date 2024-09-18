# currently it will only work for historical literature cross NER.
import re
import os
import ast
import json
import time
import pickle
import traceback
import numpy as np
from src.config import *
from tqdm.auto import tqdm
from src.utils import caching_layer
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Anyscale
from src.config import gpt_llm, claude_llm, anyscale_llm

# system_prompt = '''Your goal is to extract the following metadata from the user text:
#
# 1. Person names
# 2. Country names
# 3. Writer names
# 4. Book titles
# 5. Award names
# 6. Literary genres
# 7. Poem titles
# 8. Location names
# 9. Organization names
# 10. Event names
# 11. Other named entities that don't fit the above categories
#
#  Follow these guidelines when extracting the metadata:
#
#         - Extract the information exactly as it appears in the text.
#         - If multiple items are found for a category, separate them with commas in a list.
#         - If no information is found for a category, leave it as an empty array.
#         - Use all relevant information you find in the text.
#
#
#     Present your findings in a JSON format, using a markdown code block. Use the following structure:
#
#     ```json
#     {
#         "person": [],
#         "country": [],
#         "writer": [],
#         "else": [],
#         "book": [],
#         "award": [],
#         "literary genre": [],
#         "poem": [],
#         "location": [],
#         "organization": [],
#         "event": []
#     }
#     ```
#
#     Fill in each array with the appropriate extracted information. If no information is found for a category, leave the array empty.
#
#     Provide your complete answer in json format.'''
#


def construct_system_prompt(labels, ner_text, model_name):
    # if model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
    #     extra_string = ""
    # else:
    extra_string = "- Return as a JSON object as specified above."

    system_prompt = '''Your goal is to extract the following metadata from the text:

        {meta_data}

         Follow these guidelines when extracting the metadata:

            - Extract the information exactly as it appears in the text.
            - If multiple items are found for a category, separate them with commas in a list.
            - If no information is found for a category, leave it as an empty array.
            - Use all relevant information you find in the text.


        Present your findings in a JSON format, using a markdown code block. Use the following structure:

            ```json
            {json_format}
            ```



            Please extract all the meta data as specified above for the following text:

            {ner_text}


            - Fill in each array with the appropriate extracted information. If no information is found for a category, leave the array empty.

            - Provide your complete answer in json format as described above. 
            - Please don't add any other information.
            - The text should only mention json once
            {extra_string}

            '''

    label_string = ""

    if "else" in labels:
        index_of_else = labels.index("else")
        labels.pop(index_of_else)
        print(labels)
        labels.insert(len(labels), "else")

    for c, l in enumerate(labels):
        if l == "else":
            l = "Other named entities that don't fit the above categories"
        label_string = label_string + str(c) + "." + " " + l.capitalize() + "\n"

    labels_dict = {}
    for l in labels:
        labels_dict[l] = []

    return system_prompt.format(meta_data=label_string, json_format=str(labels_dict),
                                ner_text=ner_text, extra_string=extra_string)


def reverse_idx_search(sub, string):
    idx_beg = string.find(sub)
    if idx_beg != -1:
        s = idx_beg
        e = idx_beg + len(sub)
        string = string[:s] + " " * len(sub) + string[e:]
        return string, (s, e)
    else:
        return string, None


def get_json(text):
    pattern = r'```json(.*?)```'

    # Search for the pattern in the text
    match = re.findall(pattern, text, re.DOTALL)

    if len(match) == 0:
        pattern = r'```json\n(.*?)\n```'
        match = re.findall(pattern, text, re.DOTALL)

    if len(match) == 0:
        pattern = r"\{(.*?)\}"
        match = re.findall(pattern, text, re.DOTALL)

    if len(match) == 0:
        return {}

    all_json_data = []

    for m in match:
        json_str = m.strip().replace("\n", "")
        try:
            json_data = ast.literal_eval(json_str)
            all_json_data.append(json_data)
        except:
            continue

    if len(all_json_data) == 0:
        return {}

    # find the match with the most number of entities found!
    all_sum = []
    for one_output in all_json_data:
        all_sum.append(sum([len(value) for key, value in one_output.items()]))

    return all_json_data[np.argmax(all_sum)]


def generate_predictions(dataset, llm, labels, redis_cli, use_redis_caching, model_name,
                         temperature):
    # system_prompt = construct_system_prompt(labels=labels)
    preds = []
    counter = 0
    for d in tqdm(dataset):
        try:
            text = " ".join(d.text)

            messages = [
                (
                    "system",
                    "you are helpful AI assistant with the aim to extract JSON output. Make sure the output is in json format",
                ),
                ("human", construct_system_prompt(labels=labels, ner_text=text,
                                                  model_name=model_name)),
            ]
            # time.sleep(3)
            # output = llm.invoke(messages)
            output = caching_layer(model_name=model_name,
                                   model_temperature=temperature, message=messages,
                                   llm=llm, redis_cli=redis_cli,
                                   use_redis_caching=use_redis_caching)
            if type(output) != str:
                output = output.content

            # print(output)

            json_output = get_json(text=output)

            custom_preds = []
            for key, value in json_output.items():
                for v in value:
                    _, location = reverse_idx_search(sub=v, string=text)
                    if location:
                        # custom_preds.append({"start": location[0], "end": location[1], "label": key})
                        custom_preds.append(
                            NERMolecule(start=location[0], end=location[1], label=key,
                                        text=""))
            preds.append(NERDataPoint(text=d.text, labels=d.labels, ners=custom_preds))
            if len(custom_preds) == 0:
                counter += 1
        except:
            traceback.print_exc()
            preds.append(NERDataPoint(text=d.text, labels=d.labels, ners=[]))

    return preds


def runner(dataset, **kwargs):
    splits = kwargs["splits"]
    llm = kwargs["method"]
    r = kwargs["redis_client"]
    use_redis_caching = kwargs["use_redis_caching"]
    # datasaving_string = kwargs["dataset_string"]

    train_pred, dev_pred, test_pred = None, None, None



    if llm in gpt_llm:
        model = ChatOpenAI(
            model=llm,
            temperature=0.0,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    elif llm in anyscale_llm:
        model = Anyscale(model_name=llm, temperature=0.0)
    elif llm in claude_llm:
        model = ChatAnthropic(
            model=llm,
            temperature=0.0,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    else:
        raise NotImplementedError

    def temp(dataset, labels):
        pred = generate_predictions(dataset=dataset, llm=model, labels=labels,
                                    redis_cli=r,
                                    use_redis_caching=use_redis_caching,
                                    model_name=llm, temperature=0.0)

        return pred

    # Path("../predictions/").mkdir(parents=True, exist_ok=True)
    # fname = f"../data/generated/{datasaving_string}.pkl"



    if "train" in splits:
        train_pred = temp(dataset["train"],
                          dataset["extra"]["labels"])

    if "dev" in splits:
        dev_pred = temp(dataset["dev"],
                        dataset["extra"]["labels"])

    if "test" in splits:
        test_pred = temp(dataset["test"],
                         dataset["extra"]["labels"])

    predictions = {
        "train": train_pred,
        "test": test_pred,
        "dev": dev_pred,
        "extra": None
    }

    return predictions
