# transform this file to generic anyscale
# create another file for diabolcom
# yet another file later for ollama

import re
import os
import pickle
import random
from pathlib import Path
from tqdm.auto import tqdm
from config import NERMolecule, NERDataPoint
from src.llm_endpoints import generate_response
from .prompts import label_prompts, example_prompt

def get_completion(prompt, llm):
    return generate_response(prompt=prompt, llm=llm)


def generate_labels(labels, llm, dataset_theme):
    generate_label_prompt = label_prompts[llm]
    examples = {}
    for label in tqdm(labels):
        output = get_completion(generate_label_prompt.format(label, dataset_theme), llm)
        examples[label] = output

    generated_labels = {}
    for label, value in examples.items():
        generated_labels[label] = re.findall(r'"(.*?)"', value)

    return generated_labels


def generate_examples(generated_labels, dataset_labels, example_string, number_of_examples, llm, dataset_theme):
    all_examples = []
    counter = 0
    generate_example_prompt = example_prompt[llm]


    pbar = tqdm(total=number_of_examples)
    while len(all_examples) < number_of_examples:
        if counter > number_of_examples + 20:
            break
        counter += 1
        try:
            number_of_labels = random.randrange(1, 5)
            labels_in_focus = random.sample(dataset_labels, number_of_labels)
            ners = {}
            for l in labels_in_focus:
                ners[l] = random.sample(generated_labels[l], random.randrange(1, min(3,len(generated_labels[l]))))


            ner_string = "Named Entities: " + str(ners).replace("{", "").replace("}", "").replace(
                '''"''', "").replace("'", "")

            prompt = generate_example_prompt.format(dataset_theme, example_string, ner_string)
            llm_output = get_completion(prompt, llm)
            llm_output = llm_output.replace("\n", "")

            final_ners = []
            for label, value in ners.items():
                for v in value:
                    start, end = re.search(v, llm_output, re.IGNORECASE).span()
                    final_ners.append(NERMolecule(start=start, end=end, label=label, text=v))

            nerd_point = NERDataPoint(text=llm_output.split(" "), labels=list(generated_labels.keys()), ners=final_ners)
            all_examples.append(nerd_point)
            pbar.update(1)
        except AttributeError:
            continue

    pbar.close()

    return all_examples


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

def generate_dataset(dataset, llm, **kwargs):

    dataset_theme = dataset["extra"]["theme"]
    dataset_name = dataset["extra"]["name"]
    dataset_labels = dataset["extra"]["labels"]

    # Step 1a: Generate Labels
    Path("../generation/").mkdir(parents=True, exist_ok=True)
    if kwargs["caching"]:
        fname = f"../generation/{dataset_name}_{dataset_theme}_{llm.replace('/','_')}.pkl"
        if os.path.isfile(fname):
            generated_labels = pickle.load(open(fname, "rb"))
        else:
            generated_labels = generate_labels(labels=dataset_labels, llm=llm, dataset_theme=dataset_theme)
            pickle.dump(generated_labels, open(fname, "wb"))
    else:
        generated_labels = generate_labels(labels=dataset_labels, llm=llm, dataset_theme=dataset_theme)

    # Step 2a: Select examples
    example_string = generate_example_string(kwargs["k_shot"], dataset["test"])


    # Step 3: Generate examples - @TODO: convert this into while loop!
    if kwargs["caching"]:
        fname = f"../generation/{dataset_name}_{llm.replace('/','_')}_{kwargs['k_shot']}_{kwargs['number_of_examples']}.pkl"
        if os.path.isfile(fname):
            all_examples = pickle.load(open(fname, "rb"))
        else:
            all_examples = generate_examples(generated_labels, dataset_labels, example_string,
                                             kwargs["number_of_examples"], llm, dataset_theme)
            pickle.dump(all_examples, open(fname, "wb"))
    else:
        all_examples = generate_examples(generated_labels, dataset_labels, example_string, kwargs["number_of_examples"], llm, dataset_theme)

    print(all_examples)
    pickle.dump(all_examples, open(f"../data/generated/{dataset_name}_{llm.replace('/','_')}_{kwargs['k_shot']}_{kwargs['number_of_examples']}.pkl", "wb"))











