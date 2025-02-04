import re
from typing import List, Dict, Any

from config import NERDataPoint, NERMolecule
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, create_model

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


def extract_json_from_output(output: str) -> str:
    match = re.search(r"(\{.*\})", output, re.DOTALL)
    if not match:
        raise ValueError("No JSON object could be extracted from the output.")
    return match.group(1)


def reverse_idx_search(sub, string):
    idx_beg = string.find(sub)
    if idx_beg != -1:
        s = idx_beg
        e = idx_beg + len(sub)
        string = string[:s] + " " * len(sub) + string[e:]
        return string, (s, e)
    else:
        return string, None


def generate_dynamic_response_schema(labels: Dict[str, str]) -> BaseModel:
    fields: Dict[str, Any] = {}
    for label, description in labels.items():
        fields[label] = (List[str], Field(default=[], description=description))

    # Create and return the dynamic Pydantic model
    return create_model("DynamicResponseSchema", **fields)


def get_predictions_over_dataset(dataset, llm_adapter, split="test"):
    all_preds = []
    for i in range(len(dataset[split])):
        parsed_results = get_prediction_over_datapoint(datapoint=dataset[split][i], label=dataset['extra']['labels'],
                                                       llm_adapter=llm_adapter)
        all_preds.append(
            NERDataPoint(text=dataset[split][i].text, labels=dataset[split][i].labels, ners=parsed_results))
    return all_preds


def get_prediction_over_datapoint(datapoint, label, llm_adapter):
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
