# currently it will only work for historical literature cross NER.
import os
import time
import pickle
from src.config import *
from tqdm.auto import tqdm
from thefuzz import process
from utils import caching_layer
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Anyscale
from src.config import gpt_llm, claude_llm, anyscale_llm

# def caching_layer(model_name, model_temperature, message, llm, redis_cli,
#                   use_redis_caching):
#     if model_temperature != 0.0 or use_redis_caching is False:
#         output = llm.invoke(message)
#         if type(output) != str:
#             output = output.content
#
#         return output
#
#     composite_key = str(model_name) + str(model_temperature) + str(message)
#     cached_response = redis_cli.get(composite_key)
#     if cached_response:
#         return cached_response
#     else:
#         output = llm.invoke(message)
#         if type(output) != str:
#             output = output.content
#         redis_cli.set(composite_key, output)
#         return output


def construct_user_prompt(labels, intent_text):
    system_prompt = '''Your goal is to extract the intent from the text provided below. The intent can be the following

    {candidate_intents}

    Follow these guidelines when extracting the metadata:
    - Make sure the intent is exactly as it appears in the above list.
    - Don't add any form of reason
    - Make sure the answer is only intent and nothing else


    Provide intent for:

    {intent_text}


    Make sure the generated text is only one of the candidate intent. And do **not** provide any other reasoning
    '''

    label_string = ""
    for index, i in enumerate(labels):
        label_string = label_string + str(index + 1) + ": " + i + "\n"

    return system_prompt.format(candidate_intents=label_string, intent_text=intent_text)


def generate_predictions(dataset, llm, labels, redis_cli, use_redis_caching, model_name, temperature):
    # system_prompt = construct_system_prompt(labels=labels)
    preds = []
    for d in tqdm(dataset):
        text = d.text

        messages = [
            (
                "system",
                "You are a helpful AI assistant with aim to extract intent",
            ),
            ("human", construct_user_prompt(labels=labels, intent_text=text)),
        ]
        output = caching_layer(model_name=model_name,
                               model_temperature=temperature, message=messages,
                               llm=llm, redis_cli=redis_cli,
                               use_redis_caching=use_redis_caching)

        pred = process.extract(output, labels, limit=1)[0][0]
        preds.append(pred)

    return preds


def runner(dataset, **kwargs):
    splits = kwargs["splits"]
    llm = kwargs["method"]
    r = kwargs["redis_client"]
    use_redis_caching = kwargs["use_redis_caching"]

    # if kwargs["number_of_examples_per_intent"] != 1:
    #     postpend_string = str(kwargs["number_of_examples_per_intent"])
    # else:
    #     postpend_string = ""

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
        # if caching:
        #     if os.path.isfile(fname):
        #         pred = pickle.load(open(fname, "rb"))
        #     else:
        #         pred = generate_predictions(dataset=dataset, llm=model, labels=labels,
        #                                     redis_cli=r,
        #                                     use_redis_caching=use_redis_caching, model_name=llm, temperature=0.0)
        #         pickle.dump(pred, open(fname, "wb"))
        # else:
        pred = generate_predictions(dataset=dataset, llm=model, labels=labels,
                                    redis_cli=r,
                                    use_redis_caching=use_redis_caching, model_name=llm, temperature=0.0)

        return pred

    # Path("../predictions/").mkdir(parents=True, exist_ok=True)
    # if kwargs["generated"]:
    #     postpend_string = postpend_string + "generated" + "_" + kwargs["llm"].replace(
    #         "/", "_") + "_"

    if "train" in splits:
        # fname = f"../predictions/train_{llm.replace('/', '_')}_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        train_pred = temp(dataset["train"],
                          dataset["extra"]["labels"])

    if "dev" in splits:
        # fname = f"../predictions/dev_{llm.replace('/', '_')}_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        dev_pred = temp(dataset["dev"],
                        dataset["extra"]["labels"])

    if "test" in splits:
        # fname = f"../predictions/test_{llm.replace('/', '_')}_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        test_pred = temp(dataset["test"],
                         dataset["extra"]["labels"])

    predictions = {
        "train": train_pred,
        "test": test_pred,
        "dev": dev_pred,
        "extra": None
    }

    return predictions
