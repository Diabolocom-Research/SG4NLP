from pathlib import Path
from .crossner import get_crossner
from .generated_dataset_parser import parse_generated_dataset
from .generic_intent_parser import get_generic_intent_dataset
from .generic_text_similarity_parser import get_generic_text_similarity_dataset

def get_dataset(dataset_name: str, **kwargs):

    if kwargs["generated"]:
        return parse_generated_dataset(dataset_name, **kwargs)
    elif dataset_name.lower() in ["crossner_ai", "crossner_literature", "crossner_music", "crossner_politics",
                                "crossner_science"]:
        kwargs["datafolder"] = kwargs["datafolder"] / Path("ner_datasets")
        return get_crossner(dataset_name, **kwargs)
    elif dataset_name.lower() in ["snips", "atis"]:
        return get_generic_intent_dataset(dataset_name, **kwargs)
    elif dataset_name.lower() in ["tweet-news", "headlines"]:
        return get_generic_text_similarity_dataset(dataset_name, **kwargs)
    else:
        raise NotImplementedError