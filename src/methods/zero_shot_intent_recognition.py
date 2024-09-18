import os
import pickle
from pathlib import Path
from tqdm.auto import tqdm
from transformers import pipeline


def generate_predictions(dataset, model, candidate_labels):
    preds = []
    for item in tqdm(dataset):
        preds.append(model(item.text,
                           candidate_labels=list(candidate_labels)))

    return [t['labels'][0] for t in preds]


def runner(dataset, **kwargs):
    splits = kwargs["splits"]
    pipe = pipeline(model="facebook/bart-large-mnli")
    candidate_labels = dataset["extra"]["labels"]
    train_pred, dev_pred, test_pred = None, None, None

    Path("../predictions/").mkdir(parents=True, exist_ok=True)

    def temp(fname, dataset, model, caching):
        if caching:
            if os.path.isfile(fname):
                pred = pickle.load(open(fname, "rb"))
            else:
                pred = generate_predictions(dataset=dataset, model=model,
                                            candidate_labels=candidate_labels)
                pickle.dump(pred, open(fname, "wb"))
        else:
            pred = generate_predictions(dataset=dataset, model=model,
                                        candidate_labels=candidate_labels)

        return pred

    postpend_string = str(kwargs["number_of_examples_per_intent"])
    if kwargs["generated"]:
        postpend_string = postpend_string + "generated" + "_" + kwargs["llm"].replace(
            "/", "_") + "_"

    if "train" in splits:
        fname = f"../predictions/train_bart_intent_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        train_pred = temp(fname, dataset["train"], pipe, kwargs["caching"])

    if "dev" in splits:
        fname = f"../predictions/dev_bart_intent_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        dev_pred = temp(fname, dataset["dev"], pipe, kwargs["caching"])

    if "test" in splits:
        fname = f"../predictions/test_bart_intent_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        test_pred = temp(fname, dataset["test"], pipe, kwargs["caching"])

    predictions = {
        "train": train_pred,
        "test": test_pred,
        "dev": dev_pred,
        "extra": None
    }

    return predictions
