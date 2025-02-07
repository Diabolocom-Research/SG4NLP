import random
import re
import traceback

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from tqdm.auto import tqdm

from config import *
from config import NERDataPoint, NERMolecule
from minimal_runner import get_dataset
from utils import get_llm_adapter


def generate_labels(theme, ner_class, llm):
    system_prompt_for_label = (
        "You are a precise and detail-oriented Named Entity Recognition (NER) assistant. "
        "Your task is to generate named entities labels belonging to the specified classes.\n"
        "Instructions:\n"
        " - Return your output strictly in JSON format, following the provided schema.\n"
        " - Be creative, but keep the names/examples plausible for the given entity type.\n"
        " - Follow the theme specified by the user while generating the examples.\n"
        " - Do not include any additional commentary, explanations, or formatting.\n"
    )

    user_prompt_for_label = (
        "Generate 50 examples/labels belonging to the NER class {ner_class}.\n\n"
        "Theme: {theme}"
        "Guidelines for generating examples/labels:\n"
        " - Follow the JSON schema exactly as specified: {format_instructions}\n"
        " - Include a mix of real and fictional names.\n"
        " - Ensure diversity in the list (e.g., different cultures, time periods, etc.).\n"
        " - Don't generate examples containing quotations or apostrophe.\n"
        "Once again, you task is to generate 50 examples/labels belonging to the NER class {ner_class}. The theme of the text should be: {theme}\n\n"
    )

    parser_label = PydanticOutputParser(pydantic_object=LabelsSchema)

    label_prompt_template = PromptTemplate(
        template=user_prompt_for_label,
        input_variables=["ner_class", "theme"],
        partial_variables={"format_instructions": parser_label.get_format_instructions()}, )
    formatted_user_prompt = label_prompt_template.format(ner_class=ner_class, theme=theme)

    messages = [
        {"role": "system", "content": system_prompt_for_label},
        {"role": "user", "content": formatted_user_prompt},
    ]
    # Use the adapter to format messages as required.
    final_prompt = llm.format_messages(messages)

    output = llm.generate(final_prompt)

    json_output_str = extract_json_from_output(output)

    json_output = parser_label.parse(json_output_str)

    return json_output.examples


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


class LabelsSchema(BaseModel):
    examples: List[str] = Field(
        default_factory=list,
        description="A list of example/labels following the specified theme and NER class.")


def extract_json_from_output(output: str) -> str:
    match = re.search(r"(\{.*\})", output, re.DOTALL)
    if not match:
        raise ValueError("No JSON object could be extracted from the output.")
    return match.group(1)


def example_message(example_string, named_entity_string, dataset_theme, llm):
    def extract_output(text):
        text = text.replace("\\", "").replace("/", "")
        pattern = r'generated_text>(.*?)<generated_text'

        # Search for the pattern in the text
        match = re.search(pattern, text, re.DOTALL)

        # Extract and print the text if the pattern is found
        if match:
            extracted_text = match.group(1).strip()
            return extracted_text.replace("\n", "")
        else:
            return "No text found between <generated_text> tags."

    system_prompt_for_text_example_gen = (
        "You are a precise and detail-oriented Named Entity Recognition (NER) assistant. "
        "Your task is to generate text that incorporates all the named entities provided by the user while maintaining a natural and creative style provided in the example."

        "Instructions:\n"
        " - Return your output strictly in the format specified by the user.\n"
        " - Be creative, but include all the named entities provided by the user.\n"
        " - Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).\n"
        " - Follow the theme specified by the user while generating the text.\n"
        " - Do not include any additional commentary, explanations, or formatting.\n"
        " - Don't add quotations or apostrophe and the entity span should be exactly as provided by the user. You can change the casing, but nothing else."
        " - Do not include any other named entities."
    )

    user_prompt_for_text_example_gen = (
        "The theme of the generated text should be.\n\n"
        "Theme: {theme} \n"
        "Some examples of the generated text are \n"
        "{examples}"
        "Please generate text with following Named Entities"
        "{named_entity_string}"
        "Guidelines for generating text:\n"
        " - Create a realistic-looking sample that could be used to test name entity recognition systems.\n"
        " - Include all the named entities specified above, and make sure they don't contain quotations or apostrophe.\n"
        '''After generating the text, please provide your output in the following format:
                <generated_text>
                    Your generated text here
                </generated_text>
        \n'''
        "Once again, you task is to generate text containing above specified named entities and following the specified theme\n"
    )

    formatted_user_prompt = user_prompt_for_text_example_gen.format(theme=dataset_theme, examples=example_string,
                                                                    named_entity_string=named_entity_string)

    messages = [
        {"role": "system", "content": system_prompt_for_text_example_gen},
        {"role": "user", "content": formatted_user_prompt},
    ]
    # Use the adapter to format messages as required.
    final_prompt = llm.format_messages(messages)
    output = llm.generate(final_prompt)
    output = extract_output(output)

    return output


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

            llm_output = example_message(example_string=example_string,
                                         dataset_theme=dataset_theme,
                                         named_entity_string=ner_string,
                                         llm=llm)
            # time.sleep(1)

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


def generate_dataset(dataset, generated_dataset_params: GenerateDataset, llm_config):
    labels = dataset['extra']['labels']
    desc = dataset['extra']['desc']
    llm = get_llm_adapter(llm_config=llm_config)

    all_generated_labels = {}
    for label, label_desc in dataset['extra']['labels'].items():
        all_generated_labels[label] = generate_labels(theme=dataset['extra']['desc'], ner_class=label, llm=llm)

    example_string = generate_example_string(k_shot=5, dataset=dataset["train"])

    examples = generate_examples(generated_labels=all_generated_labels, dataset_labels=labels, example_string=example_string,
                      number_of_examples=generated_dataset_params.number_of_examples_to_generate, llm=llm,
                      dataset_theme=desc)

    generated_dataset = {
        "train": None,
        "dev": None,
        "test": examples,
        "extra": {
            "desc": desc,
            "labels": labels,
            "name": dataset["extra"]["name"] + "_generated",
            "theme": dataset["extra"]["theme"]
        }
    }

    return generated_dataset

if __name__ == "__main__":
    dataset_params = Dataset(name="crossner_politics", number_of_test_examples=50)
    generated_dataset_params = GenerateDataset()
    llm_config = LLMConfig()
    method_params = MethodArguments()
    benchmark_params = MinimalBenchmarkArguments()

    dataset = get_dataset(dataset_params=dataset_params, generated_dataset_params=generated_dataset_params)
    generate_dataset(dataset, generated_dataset_params, llm_config)

    # # get the labels and desc
    # labels = dataset['extra']['labels']
    # desc = dataset['extra']['desc']
    #
    # llm = get_llm_adapter(llm_config=llm_config)
    #
    # all_generated_labels = {}
    # for label, label_desc in dataset['extra']['labels'].items():
    #     all_generated_labels[label] = generate_labels(theme=dataset['extra']['desc'], ner_class=label, llm=llm)
