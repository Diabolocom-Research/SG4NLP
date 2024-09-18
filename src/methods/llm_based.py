import re
import os
import pickle
import requests
import traceback
from src.config import *
from pathlib import Path
from tqdm.auto import tqdm
from dotenv import load_dotenv
from .prompts import get_entities_prompt
from langchain.prompts import PromptTemplate
from .common_utils import get_response_schema
from src.llm_endpoints import generate_response
from langchain.output_parsers import StructuredOutputParser




def get_completion(prompt, llm):
    return generate_response(prompt=prompt, llm=llm)

def reverse_idx_search(sub, string):
    idx_beg = string.find(sub)
    if idx_beg!=-1:
        s = idx_beg
        e = idx_beg+len(sub)
        string = string[:s] + " "*len(sub) + string[e:]
        return string, (s, e)
    else:
        return string, None

def generate_predictions(dataset, metadata_output_schema_parser, llm):
    metadata_output_schema = metadata_output_schema_parser.get_format_instructions()

    metadata_prompt_template = PromptTemplate.from_template(
        template=get_entities_prompt[llm])

    preds = []
    for d in tqdm(dataset):
        try:
            text = " ".join(d.text)
            conversation_metadata_recognition_prompt = (
                metadata_prompt_template.format(
                    chat_history=text,
                    format_instructions=metadata_output_schema
                )
            )

            metadata_detected_str = get_completion(conversation_metadata_recognition_prompt,
                                                                llm=llm)

            metadata_detected_str = re.findall(r"\{[\s\S]+\}", metadata_detected_str)[0]
            metadata_detected = metadata_output_schema_parser.parse(
                metadata_detected_str)

            custom_preds = []
            for key, value in metadata_detected.items():
                for v in value:
                    _, location = reverse_idx_search(sub=v, string=text)
                    if location:
                        # custom_preds.append({"start": location[0], "end": location[1], "label": key})
                        custom_preds.append(NERMolecule(start=location[0], end=location[1], label=key, text=""))
            preds.append(NERDataPoint(text=d.text, labels=d.labels, ners=custom_preds))
        except:
            traceback.print_exc()
            preds.append(NERDataPoint(text=d.text, labels=d.labels, ners=[]))

    return preds

def runner(dataset, **kwargs):
    splits = kwargs["splits"]
    llm = kwargs["method"]

    train_pred, dev_pred, test_pred = None, None, None

    metadata_output_schema_parser = StructuredOutputParser.from_response_schemas(
        get_response_schema(labels=dataset["test"][0].labels)
    )



    def temp(fname, dataset, caching):
        if caching:
            if os.path.isfile(fname):
                pred = pickle.load(open(fname, "rb"))
            else:
                pred = generate_predictions(dataset=dataset, metadata_output_schema_parser=metadata_output_schema_parser, llm=llm)
                pickle.dump(pred, open(fname, "wb"))
        else:
            pred = generate_predictions(dataset=dataset, metadata_output_schema_parser=metadata_output_schema_parser, llm=llm)

        return pred

    Path("../predictions/").mkdir(parents=True, exist_ok=True)
    postpend_string = ""
    if kwargs["generated"]:
        postpend_string = postpend_string + "generated" + "_" + kwargs["llm"].replace("/", "_") + "_"

    if "train" in splits:
        fname = f"../predictions/train_{llm.replace('/','_')}_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        train_pred = temp(fname, dataset["train"], kwargs["caching"])

    if "dev" in splits:
        fname = f"../predictions/dev_{llm.replace('/','_')}_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        dev_pred = temp(fname, dataset["dev"], kwargs["caching"])

    if "test" in splits:
        fname = f"../predictions/test_{llm.replace('/','_')}_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        test_pred = temp(fname, dataset["test"], kwargs["caching"])

    predictions = {
        "train": train_pred,
        "test": test_pred,
        "dev": dev_pred,
        "extra": None
    }

    return predictions
