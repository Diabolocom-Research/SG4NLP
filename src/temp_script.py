import re
import json
import redis
from functools import partial
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, create_model
from transformers import AutoTokenizer
from llms import llm_creation # custom module for internal LLM endpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from pathlib import Path
from langchain.output_parsers import ResponseSchema
from dataclasses import dataclass
from nervaluate import Evaluator

from pprint import pprint

from config import NERDataPoint, NERMolecule

# -----------------------------------------------------------------------------
# 4. Prompt and Parser Utilities
# -----------------------------------------------------------------------------

def generate_dynamic_response_schema(labels: List[str]) -> BaseModel:
    """
    Generates a dynamic Pydantic model for a given list of label names.

    Each label becomes a field of type List[str] with a default empty list and
    a descriptive message indicating that the extraction should return an array.

    Args:
        labels (List[str]): A list of label names (e.g., "person", "country").

    Returns:
        A Pydantic BaseModel dynamically created with the specified fields.
    """
    # Default descriptions for known labels
    default_descriptions: Dict[str, str] = {
        "person": "Extract all person names in an array. Return an empty array if none are found.",
        "country": "Extract all country names in an array. Return an empty array if none are found.",
        "writer": "Extract all writer names in an array. Return an empty array if none are found.",
        "book": "Extract all book titles in an array. Return an empty array if none are found.",
        "award": "Extract all award names in an array. Return an empty array if none are found.",
        "literary genre": "Extract all literary genres in an array. Return an empty array if none are found.",
        "poem": "Extract all poem titles in an array. Return an empty array if none are found.",
        "location": "Extract all location names in an array. Return an empty array if none are found.",
        "event": "Extract all event names in an array. Return an empty array if none are found.",
        "organization": "Extract all organization names in an array. Return an empty array if none are found.",
        "magazine": "Extract all magazine names in an array. Return an empty array if none are found.",
        "political party": "Extract all political party names in an array, capturing their exact mentions. Return an empty array if none are found.",
        "election": "Extract all election references (e.g., election names or identifiers) in an array. Return an empty array if none are found.",
        "else": "Extract any named entities that do not fit the above categories in an array. Return an empty array if none are found."
    }

    descriptions_crossner_political = {
        "election": "Extract all election references (e.g., election names or identifiers) in an array. Return an empty array if none are found.",
        "else": "Extract any named entities that do not fit the specified categories in an array. Return an empty array if none are found.",
        "political party": "Extract all political party names in an array, capturing their exact mentions. Return an empty array if none are found.",
        "organization": "Extract all organization names in an array, preserving their original text. Return an empty array if none are found.",
        "politician": "Extract all names of individuals recognized as politicians in an array. Return an empty array if none are found.",
        "person": "Extract all person names in an array, preserving their exact appearance in the text. Return an empty array if none are found.",
        "event": "Extract all event names in an array, capturing their precise text. Return an empty array if none are found.",
        "country": "Extract all country names in an array, preserving the original text. Return an empty array if none are found.",
        "location": "Extract all location names in an array, exactly as they appear in the text. Return an empty array if none are found."
    }

    fields: Dict[str, Any] = {}
    for label in labels:
        description = descriptions_crossner_political.get(
            label,
            f"Extract all {label} values in an array. Return an empty array if none are found."
        )
        fields[label] = (List[str], Field(default=[], description=description))

    # Create and return the dynamic Pydantic model
    return create_model("DynamicResponseSchema", **fields)


def extract_json_from_output(output: str) -> str:
    match = re.search(r"(\{.*\})", output, re.DOTALL)
    if not match:
        raise ValueError("No JSON object could be extracted from the output.")
    return match.group(1)

# -----------------------------------------------------------------------------
# 5. Prompt Templates
# -----------------------------------------------------------------------------

# System prompt remains generic across LLMs.
system_prompt = (
    "You are a precise and detail-oriented Named Entity Recognition (NER) assistant. "
    "Your task is to extract named entities from the provided text based solely on the specified classes.\n"
    "Instructions:\n"
    " - Return your output strictly in JSON format, following the provided schema.\n"
    " - Each key (entity class) should map to an array of strings that exactly match the occurrences in the text.\n"
    " - Do not include any additional commentary, explanations, or formatting.\n"
    " - If no entity is found for a class, return an empty array for that class."
)

user_prompt = (
    "Extract the named entities from the text below.\n\n"
    "Guidelines:\n"
    " - Follow the JSON schema exactly as specified: {format_instructions}\n"
    " - For each provided entity class, return an array of entity mentions as they appear in the text.\n"
    " - Do not include any additional text or explanations.\n"
    " - If an entity class is not present, return an empty array.\n\n"
    "Text: {text}"
)

# -----------------------------------------------------------------------------
# 6. Main Flow
# -----------------------------------------------------------------------------

def main(text: str, llm_type: str = "diabolocom"):
    # Configuration can come from a config file or environment variables.
    config = {
        "model_name": "llama3",
        "redis_host": "localhost",
        "redis_port": 6379,
        "project_string": "test",
        "temperature": 0.0,
    }

    # Create LLM instance using the factory.
    llm_adapter = llm_creation.LLMFactory.create_llm(llm_type, config)

    # Create dynamic schema for the desired labels.
    DynamicSchema = generate_dynamic_response_schema(["person", "event"])
    parser_ner = PydanticOutputParser(pydantic_object=DynamicSchema)

    # Prepare the prompt.
    prompt_template = PromptTemplate(
        template=user_prompt,
        input_variables=["text"],
        partial_variables={"format_instructions": parser_ner.get_format_instructions()},
    )
    formatted_user_prompt = prompt_template.format(text=text)

    # Build messages.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_user_prompt},
    ]
    # Use the adapter to format messages as required.
    final_prompt = llm_adapter.format_messages(messages)

    # Generate output from the LLM.
    output = llm_adapter.generate(final_prompt)

    try:
        json_output_str = extract_json_from_output(output)
        parsed_result = parser_ner.parse(json_output_str)
        print("Parsed Result:", parsed_result)
    except Exception as e:
        print("Error parsing LLM output:", e)
        print("Full LLM Output:", output)



from parse_datasets import dataset_parser

def get_dataset():

    DATA_FOLDER = Path("../data")
    dataset_name = "crossner_politics"

    dataset_params = {
        "datafolder": DATA_FOLDER,
        "generated": False,
        "k_shot": None,
        "llm": None,
        "number_of_examples": 100
    }


    dataset = dataset_parser.get_dataset(dataset_name, **dataset_params)
    return dataset


def reverse_idx_search(sub, string):
    idx_beg = string.find(sub)
    if idx_beg!=-1:
        s = idx_beg
        e = idx_beg+len(sub)
        string = string[:s] + " "*len(sub) + string[e:]
        return string, (s, e)
    else:
        return string, None



def get_prediction(datapoint, label, llm_adapter):
    text = " ".join(datapoint.text)

    DynamicSchema = generate_dynamic_response_schema(label)
    parser_ner = PydanticOutputParser(pydantic_object=DynamicSchema)

    # Prepare the prompt.
    prompt_template = PromptTemplate(
        template=user_prompt,
        input_variables=["text"],
        partial_variables={"format_instructions": parser_ner.get_format_instructions()},
    )
    formatted_user_prompt = prompt_template.format(text=text)

    # Build messages.
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_user_prompt},
    ]
    # Use the adapter to format messages as required.
    final_prompt = llm_adapter.format_messages(messages)

    output = llm_adapter.generate(final_prompt)
    custom_preds = []

    try:
        json_output_str = extract_json_from_output(output)
        parsed_result = parser_ner.parse(json_output_str)
        print("Parsed Result:", parsed_result)
        for key, value in parsed_result.dict().items():
            for v in value:
                _, location = reverse_idx_search(sub=v, string=text)
                if location:
                    # custom_preds.append({"start": location[0], "end": location[1], "label": key})
                    custom_preds.append(NERMolecule(start=location[0], end=location[1], label=key, text=""))
        return custom_preds
    except Exception as e:
        print("Error parsing LLM output:", e)
        print("Full LLM Output:", output)
        return custom_preds





if __name__ == "__main__":
    sample_text = "Hello! I am Donald Trump, and I am running for the American Election?"
    llm_type = "diabolocom"
    # main(sample_text)
    dataset = get_dataset()

    config = {
        "model_name": "llama3",
        "redis_host": "localhost",
        "redis_port": 6379,
        "project_string": "test",
        "temperature": 0.0,
    }

    # Create LLM instance using the factory.
    llm_factory = llm_creation.LLMFactory()
    llm_adapter = llm_factory.create_llm(llm_type, config)

    all_preds = []
    for i in range(100):
        parsed_results = get_prediction(datapoint = dataset["train"][i], label=dataset['extra']['labels'], llm_adapter=llm_adapter)
        all_preds.append(NERDataPoint(text=dataset["train"][i].text, labels=dataset["train"][i].labels, ners=parsed_results))
        # input()

    evaluator = Evaluator([[i.__dict__ for i in d.ners] for d in dataset["train"][:100]],
                          [[i.__dict__ for i in d.ners] for d in all_preds],
                          tags=dataset["test"][0].labels)

    pprint(evaluator.evaluate())

    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()







