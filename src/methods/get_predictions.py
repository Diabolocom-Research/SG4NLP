from .gliner import runner as gliner_runner
from .llm_based import runner as mixtral_runner
from .llm_based_langchain_ner import runner as langchain_ner
from .llm_based_langchain_intent import runner as langchain_intent
from .zero_shot_intent_recognition import runner as zero_shot_runner
from .llm_based_langchain_text_similarity import runner as langchain_text_similarity

from src.config import big_llm_list


def predictions(method_name, dataset, **kwargs):
    if method_name == "gliner" and kwargs["task"] == "ner":
        return gliner_runner(dataset, **kwargs)
    elif method_name in ["llama3:70b-instruct-q4"] and kwargs["task"] == "ner":
        return mixtral_runner(dataset, **kwargs)
    elif method_name in big_llm_list and kwargs["task"] == "ner":
        return langchain_ner(dataset, **kwargs)
    elif method_name in ["huggingface_bart_large"] and kwargs["task"] == "intent":
        return zero_shot_runner(dataset, **kwargs)
    elif method_name in big_llm_list and kwargs["task"] == "intent":
        return langchain_intent(dataset, **kwargs)
    elif method_name in big_llm_list and kwargs[
        "task"] == "text_similarity":
        return langchain_text_similarity(dataset, **kwargs)
    else:
        raise NotImplementedError
