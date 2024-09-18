from pathlib import Path
from config import IntentDataPoint

def return_dataset(path):
    label = open(path/ Path("label"), "r").read()
    label = label.split("\n")

    text = open(path/ Path("seq.in"), "r").read()
    text = text.split("\n")

    dataset = []

    for l,t in zip(label, text):
        if l != '' and text != '':
            # intent_datapoint = IntentDataPoint(label=" ".join(l.split("_")), text=t)
            intent_datapoint = IntentDataPoint(label=l, text=t)
            dataset.append(intent_datapoint)

    return dataset
def get_generic_intent_dataset(dataset_name: str, **kwargs):
    datafolder = kwargs['datafolder']
    dataset_path = Path(datafolder) / Path("intent_datasets")/Path(dataset_name)

    train_dataset = return_dataset(dataset_path/Path("train"))
    dev_dataset = return_dataset(dataset_path / Path("valid"))
    test_dataset = return_dataset(dataset_path / Path("test"))

    all_labels = sorted(set([t.label for t in train_dataset]))


    return {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset[:kwargs["number_of_examples_for_original_dataset"]],
        "extra": {"name": dataset_name,
                  "labels": list(all_labels),
                  "theme": ""}
    }