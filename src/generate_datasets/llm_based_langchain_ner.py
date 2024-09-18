import re
import os
import json
import time
import pickle
import random
import requests
import traceback
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from config import NERMolecule, NERDataPoint
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Anyscale
from src.config import gpt_llm, claude_llm, anyscale_llm


def label_message(dataset_theme, label_type):
    prompt = '''You are tasked with generating 50 examples of a specific type of label. Your goal is to create a diverse list that includes both real and fictional examples, some of which may have {theme} significance.

    Guidelines for generating names/examples:
    - Include a mix of real and fictional names
    - Some examples should have potential {theme} significance
    - Ensure diversity in the list (e.g., different cultures, time periods, etc.)
    - Be creative, but keep the names/examples plausible for the given person type
    - Do not add any extra information beyond the names/examples themselves


    Please provide your output as an array of 50 names/examples. Use the following format:

    <output>
    [
      "Example 1",
      "Example 2",
      "Example 3",
      ...
      "Example 50"
    ]
    </output>


    The type of label you should generate names/examples for is:

    <label_type>
    {label_type}
    </label_type>

    - Remember to include entries in the array, separated by commas, and enclosed in square brackets. Do not include any additional information or explanations outside of the array.
    - Make up examples if you don't know about it.
    - Remember to include each example in a double quote

    '''

    prompt = prompt.format(theme=dataset_theme, label_type=label_type)

    messages = [
        (
            "system",
            "You are a helpful assistant who follows user request.",
        ),
        ("human", prompt)
    ]

    return messages


def example_message(example_string, dataset_theme, named_entity_string):
    _system_prompt = '''You are an AI assistant tasked with generating a short text sample for evaluating name entity recognition systems. Your goal is to create a text that incorporates all of the provided named entities provided by the user while maintaining a natural and creative style provided in the example. Some examples are

{example_string}


Follow these instructions to generate the text:

1. Create a short text that incorporates ALL of the provided named entities.
2. Ensure the text follows the theme similar to examples described above.
3. Be creative and keep the text natural, following the style and tone of the examples provided in the original task description.
4. Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).
5. Make absolutely sure that all named entities are mentioned in the text.
6. Don't add quotations or apostrophe.
7. Do not include any other named entities

When writing your text:
- Use proper capitalization and punctuation.
- Vary sentence structure to maintain a natural flow.
- Create logical connections between the named entities to form a coherent narrative.
- Aim for a length and theme similar to the examples above.

After generating the text, please provide your output in the following format:

<generated_text>
[Your generated text here]
</generated_text>


Remember, the goal is to create a realistic-looking sample that could be used to test name entity recognition systems. Make sure your text sounds natural and incorporates all the named entities exactly as given by the user.'''

    # prompt = prompt.format(example_string=example_string,
    #                        dataset_theme=dataset_theme,
    #                        named_entity_string=named_entity_string)

    user_prompt = '''{named_entity_string}'''

    messages = [
        (
            "system",
            _system_prompt.format(example_string=example_string,
                                  dataset_theme=dataset_theme)
        ),
        ("human", user_prompt.format(named_entity_string=named_entity_string))
    ]

    # pprint(messages)

    return messages


def generate_labels(labels, llm, dataset_theme):
    examples = {}
    for label in tqdm(labels):
        message = label_message(dataset_theme=dataset_theme, label_type=label)
        output = llm.invoke(message)
        # print(output)
        if type(output) == str:
            examples[label] = output
        else:
            examples[label] = output.content

    generated_labels = {}
    for label, value in examples.items():
        generated_labels[label] = re.findall(r'"(.*?)"', value)

    generated_labels[label] = [i for i in generated_labels[label] if i.strip() != ""]

    return generated_labels


def generate_example_string(k_shot, dataset):
    number_of_examples_to_show = k_shot
    indexes = random.sample(range(len(dataset)), number_of_examples_to_show)

    final_string = ""
    for c, index in enumerate(indexes):
        data_point = dataset[index]
        text = " ".join(data_point.text)
        ner_dict = {}
        for ner in data_point.ners:
            temp = ner_dict.get(ner.label, [])
            temp.append(ner.text.strip())
            ner_dict[ner.label] = temp
        _final_string = f"Example {c}" + "\n"
        _final_string = _final_string + "Named Entities: " + str(ner_dict).replace("{", "").replace("}", "").replace(
            '''"''', "").replace("'", "") + "\n"
        _final_string = _final_string + "Text:" + text + "\n"
        final_string = final_string + _final_string + "\n"

    return final_string


def clean_output(text):
    pattern = r'generated_text>(.*?)</generated_text'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # Extract and print the text if the pattern is found
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        return "No text found between <generated_text> tags."

def generate_examples(generated_labels, dataset_labels, example_string,
                      number_of_examples, llm, dataset_theme):
    all_examples = []
    counter = 0

    pbar = tqdm(total=number_of_examples)
    while len(all_examples) < number_of_examples:
        if counter > number_of_examples + 400:
            break
        counter += 1
        try:
            number_of_labels = random.randrange(1, 5)
            temp_labels = [key for key, value in generated_labels.items() if
                           len(value) > 0]
            labels_in_focus = random.sample(temp_labels, number_of_labels)
            ners = {}
            for l in labels_in_focus:
                ners[l] = random.sample(generated_labels[l], random.randrange(1, min(3,
                                                                                     len(
                                                                                         generated_labels[
                                                                                             l]))))

            ner_string = "Named Entities: " + str(ners).replace("{", "").replace("}",
                                                                                 "").replace(
                '''"''', "").replace("'", "")

            # print(ner_string)

            prompt = example_message(example_string=example_string,
                                     dataset_theme=dataset_theme,
                                     named_entity_string=ner_string)
            # time.sleep(1)
            llm_output = llm.invoke(prompt)


            if type(llm_output) != str:
                llm_output = llm_output.content

            # print(llm_output)
            # print("**")
            llm_output = clean_output(llm_output)

            llm_output = llm_output.replace("\n", "")

            final_ners = []
            for label, value in ners.items():
                for v in value:
                    start, end = re.search(v, llm_output, re.IGNORECASE).span()
                    final_ners.append(
                        NERMolecule(start=start, end=end, label=label, text=v))

            nerd_point = NERDataPoint(text=llm_output.split(" "),
                                      labels=list(generated_labels.keys()),
                                      ners=final_ners)
            all_examples.append(nerd_point)
            pbar.update(1)
        except:
            traceback.print_exc()
            continue

    pbar.close()

    return all_examples



def generate_dataset(dataset, llm, **kwargs):
    dataset_theme = dataset["extra"]["theme"]
    dataset_name = dataset["extra"]["name"]
    dataset_labels = dataset["extra"]["labels"]
    datasaving_string = kwargs["dataset_string"]
    k = kwargs["k_shot"]

    temperature = 0.99
    if llm in gpt_llm:
        model = ChatOpenAI(
            model=llm,
            temperature=temperature,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    elif llm in anyscale_llm:
        model = Anyscale(model_name=llm, temperature=temperature)
    elif llm in claude_llm:
        model = ChatAnthropic(
            model=llm,
            temperature=temperature,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    else:
        raise NotImplementedError


    # Step 1a: Generate Labels
    # Path("../generation/").mkdir(parents=True, exist_ok=True)
    fname = f"../data/generated/{datasaving_string}.pkl"
    fname_for_labels = f"../data/generated/{datasaving_string}_labels.pkl"



    if kwargs["caching"]:
        # fname = f"../generation/{dataset_name}_{dataset_theme}_{llm.replace('/','_')}.pkl"
        if os.path.isfile(fname_for_labels):
            generated_labels = pickle.load(open(fname_for_labels, "rb"))
        else:
            generated_labels = generate_labels(labels=dataset_labels, llm=model, dataset_theme=dataset_theme)
            pickle.dump(generated_labels, open(fname_for_labels, "wb"))
    else:
        generated_labels = generate_labels(labels=dataset_labels, llm=model, dataset_theme=dataset_theme)

    # Step 2a: Select examples
    example_string = generate_example_string(kwargs["k_shot"], dataset["train"])


    # Step 3: Generate examples
    if kwargs["caching"]:
        # fname = f"../generation/{dataset_name}_{llm.replace('/','_')}_{kwargs['k_shot']}_{kwargs['number_of_examples']}.pkl"
        if os.path.isfile(fname):
            all_examples = pickle.load(open(fname, "rb"))
        else:
            all_examples = generate_examples(generated_labels, dataset_labels, example_string,
                                             kwargs["number_of_examples"], model, dataset_theme)
            pickle.dump(all_examples, open(fname, "wb"))
    else:
        all_examples = generate_examples(generated_labels, dataset_labels, example_string, kwargs["number_of_examples"], model, dataset_theme)

    print(all_examples)
    pickle.dump(all_examples, open(fname, "wb"))



