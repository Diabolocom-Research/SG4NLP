import src.generate_datasets.llm_based as mixtral
import src.generate_datasets.llm_based_langchain_ner as llm_langchain_ner
import src.generate_datasets.llm_based_langchain_intent as llm_langchain_intent
import src.generate_datasets.llm_based_langchain_text_similarity as llm_text_similarity
from src.config import big_llm_list
def router(llm, dataset, **kwargs):




    if llm in ["llama3:70b-instruct-q4"] and kwargs["task"] == "ner":
        mixtral.generate_dataset(dataset, llm, **kwargs)
    elif llm in big_llm_list  and kwargs["task"] == "ner":
        llm_langchain_ner.generate_dataset(dataset, llm, **kwargs)
    elif llm in big_llm_list  and kwargs["task"] == "intent":
        llm_langchain_intent.generate_dataset(dataset, llm, **kwargs)
    elif llm in big_llm_list and kwargs["task"] == "text_similarity":
        llm_text_similarity.generate_dataset(dataset, llm, **kwargs)
    else:
        print(f"the llm is {llm}")
        raise NotImplementedError
