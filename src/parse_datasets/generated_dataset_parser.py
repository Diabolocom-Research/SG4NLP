import pickle

def parse_generated_dataset(dataset_name, **kwargs):
    llm = kwargs["llm"]
    context_file_name = kwargs["dataset_string"]
    file_name = f"../data/generated/{context_file_name}.pkl"


    examples = pickle.load(open(file_name, "rb"))



    if kwargs["task"] == "ner":
        return {
            "train": None,
            "dev": None,
            "test": examples,
            "extra": {"theme": None,
                      "name": dataset_name,
                      "file_name": file_name,
                      "labels": examples[0].labels}
        }

    elif kwargs["task"] == "intent":
        return {
            "train": None,
            "dev": None,
            "test": examples,
            "extra": {"theme": None,
                      "name": dataset_name,
                      "file_name": file_name,
                      "labels": sorted(list(set(e.label for e in examples)))}
        }
    elif kwargs["task"] == "text_similarity":
        return {
            "train": None,
            "dev": None,
            "test": examples,
            "extra": {"theme": None,
                      "name": dataset_name,
                      "file_name": file_name,
                      "labels": list(set(e.score for e in examples))}
        }
    else:
        raise NotImplementedError
