import pandas as pd
from pathlib import Path
from config import TextSimilarityDataPoint

def return_dataset(path):
    df = pd.read_csv(path, delimiter="\t")

    dataset = []
    for i in df.iterrows():
        tsdp = TextSimilarityDataPoint(text1=i[1][1], text2=i[1][2], score=round(i[1][0]))
        dataset.append(tsdp)

    return dataset
def get_generic_text_similarity_dataset(dataset_name: str, **kwargs):
    datafolder = kwargs['datafolder']
    if dataset_name == "headlines":
        dataset_path = Path(datafolder) / Path("sts_datasets")/Path("headlines.test.tsv")
        theme = "news headlines"
    elif dataset_name == "tweet-news":
        dataset_path = Path(datafolder) / Path("sts_datasets") / Path(
            "tweet-news.test.tsv")
        theme = "tweet news headline"
    else:
        raise NotImplementedError

    train_dataset = None
    dev_dataset = None
    test_dataset = return_dataset(dataset_path)[:kwargs["number_of_examples_for_original_dataset"]]

    all_scores = set([t.score for t in test_dataset])

    return {
        "train": train_dataset,
        "dev": dev_dataset,
        "test": test_dataset,
        "extra": {"name": dataset_name,
                  "labels": list(all_scores),
                  "theme": theme}
    }