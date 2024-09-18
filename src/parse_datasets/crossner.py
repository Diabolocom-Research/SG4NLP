from pathlib import Path
from .common_utils import *
from config import NERDataPoint, NERMolecule


def reformat_dataset(dataset, labels):
    reformated_dataset = []
    for t in dataset:
        all_ner = flatten_ner(t["tokenized_text"], t["ner"])
        ners = []
        for ner in all_ner:
            ner_mol = NERMolecule(**ner)
            ners.append(ner_mol)
        ner_data_point = NERDataPoint(text=t["tokenized_text"], ners=ners, labels=labels)
        reformated_dataset.append(ner_data_point)

    return reformated_dataset


def get_crossner(dataset_name: str, **kwargs):
    """Parses and generate crossner dataset in a specific format"""
    dataset_name = dataset_name.lower()
    name_mapper = {
        "CrossNER_AI": "crossner_ai",
        "CrossNER_literature": "crossner_literature",
        "CrossNER_music": "crossner_music",
        "CrossNER_politics": "crossner_politics",
        "CrossNER_science": "crossner_science"
    }
    name_mapper = {value: key for key, value in name_mapper.items()}

    datafolder = kwargs['datafolder']
    dataset_path = Path(datafolder) / Path("cross_ner") / Path(name_mapper[dataset_name])
    train_dataset, dev_dataset, test_dataset, labels = create_dataset(dataset_path)
    train_dataset = reformat_dataset(train_dataset, labels)
    dev_dataset = reformat_dataset(dev_dataset, labels)
    test_dataset = reformat_dataset(test_dataset, labels)


    if kwargs["number_of_ner"]:
        k = kwargs["number_of_examples"]
        custom_split = []
        for i in test_dataset+dev_dataset+train_dataset:
            if kwargs["number_of_ner"][0] <= len(i.ners) <= kwargs["number_of_ner"][1]:
                custom_split.append(i)
            if len(custom_split) == k:
                break

        test_dataset = custom_split

    return {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset[:kwargs["number_of_examples_for_original_dataset"]],
        "extra": {"theme": dataset_name.split("_")[1],
                  "name": dataset_name,
                  "labels": labels}
    }
