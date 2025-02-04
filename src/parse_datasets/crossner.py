from pathlib import Path
from .common_utils import *
from config import NERDataPoint, NERMolecule
from .crossner_utils import *

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

    desc_labels = {
        "crossner_ai": crossner_ai_labels_and_desc(),
        "crossner_literature": crossner_lit_labels_and_desc(),
        "crossner_music": crossner_music_labels_and_desc(),
        "crossner_politics": crossner_politics_labels_and_desc(),
        "crossner_science": crossner_natural_science_labels_and_desc()
    }

    common_label = common_labels()

    name_mapper = {value: key for key, value in name_mapper.items()}

    datafolder = kwargs['datafolder']
    dataset_path = Path(datafolder) / Path("cross_ner") / Path(name_mapper[dataset_name])
    train_dataset, dev_dataset, test_dataset, labels = create_dataset(dataset_path)
    train_dataset = reformat_dataset(train_dataset, labels)
    dev_dataset = reformat_dataset(dev_dataset, labels)
    test_dataset = reformat_dataset(test_dataset, labels)

    desc, dataset_specific_labels = desc_labels[dataset_name]

    final_labels = {}
    for label in labels:
        if label in common_label:
            final_labels[label] = common_label[label]
        else:
            final_labels[label] = dataset_specific_labels[label]

    return {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset,
        "extra": {"theme": dataset_name.split("_")[1],
                  "name": dataset_name,
                  "labels": final_labels,
                  "desc": desc
                  }
    }
