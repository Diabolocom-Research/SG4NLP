import os
import pickle
from src.config import *
from gliner import GLiNER
from tqdm.auto import tqdm

def generate_predictions(dataset, model:GLiNER):
    preds = []
    for d in tqdm(dataset):
        ent = model.predict_entities(" ".join(d.text), labels=d.labels)
        ent = [NERMolecule(**e) for e in ent]
        preds.append(NERDataPoint(text=d.text, labels=d.labels, ners=ent))

    return preds


def runner(dataset, **kwargs):
    splits = kwargs["splits"]
    model = GLiNER.from_pretrained("urchade/gliner_largev2")
    train_pred, dev_pred, test_pred = None, None, None

    if kwargs["k_shot"] == 5 and kwargs["k_shot"] == 200:
        postpend_string = ""
    else:
        postpend_string = str(kwargs["k_shot"]) + "_" + str(
            kwargs["number_of_examples"]) + "_"

    def temp(fname, dataset, model, caching):
        if caching:
            if os.path.isfile(fname):
                pred = pickle.load(open(fname, "rb"))
            else:
                pred = generate_predictions(dataset=dataset, model=model)
                pickle.dump(pred, open(fname, "wb"))
        else:
            pred = generate_predictions(dataset=dataset, model=model)

        return pred

    Path("../predictions/").mkdir(parents=True, exist_ok=True)


    if kwargs["generated"]:
        postpend_string = postpend_string + "generated" + "_" + kwargs["llm"].replace("/", "_") + "_"

    if "train" in splits:
        fname = f"../predictions/train_gliner_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        train_pred = temp(fname, dataset["train"], model, kwargs["caching"])

    if "dev" in splits:
        fname = f"../predictions/dev_gliner_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        dev_pred = temp(fname, dataset["dev"], model, kwargs["caching"])

    if "test" in splits:
        fname = f"../predictions/test_gliner_zero_shot_{kwargs['dataset_name']}_{postpend_string}.pkl"
        test_pred = temp(fname, dataset["test"], model, kwargs["caching"])

    predictions = {
        "train": train_pred,
        "test": test_pred,
        "dev": dev_pred,
        "extra": None
    }

    return predictions




