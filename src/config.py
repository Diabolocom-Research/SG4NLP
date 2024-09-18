'''
The file provides data class specifing interface for input/output
'''

from pathlib import Path
from typing import List, Optional
from pydantic.dataclasses import dataclass

DATA_FOLDER = Path("../data")
MLFLOW_URI = "http://127.0.0.1:8080"

llm_lookup = [
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "anyscale"),
    ("llama3:70b-instruct-q4", "diabolocom")
]

anyscale_llm = ["mistralai/Mixtral-8x7B-Instruct-v0.1",
                "meta-llama/Meta-Llama-3-70B-Instruct",
                "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "google/gemma-7b-it"
                ]

gpt_llm = ["gpt-4o", "gpt-3.5-turbo", "gpt-4o-mini"]

claude_llm = ["claude-3-5-sonnet-20240620", "claude-3-haiku-20240307"]

big_llm_list = gpt_llm + claude_llm + anyscale_llm


@dataclass
class BenchmarkRunnerArguments:
    """Input arguments for the benchmark runner i.e files which runs the llm over the task"""
    use_mlflow: bool = False # Starts the ML flow logging
    run_id: int = 0
    dataset: str = "crossner_literature"
    method: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # huggingface_bart_large, mistralai/Mixtral-8x7B-Instruct-v0.1
    split: str = "test"
    caching: bool = False
    llm: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    k_shot: int = 5
    number_of_examples: int = 10
    generated: bool = True # specifies if the user wants to use the generated data or original data
    seed: int = 42
    task: str = "ner"
    number_of_examples_per_intent: int = 5
    use_redis_caching: bool = True # We are using reddis to cache responses from the llm to save some cost
    port: int = 6379
    number_of_ner: Optional[set] = None  # Specifies number of examples used for generating ner examples
    number_of_examples_per_score: int = 5 # Specifies number of examples used per score generating text similarity examples
    number_of_examples_for_original_dataset: int = 10 # Specifies number of examples used per score generating text similarity examples


@dataclass
class GenerateRunnerArguments:
    run_id: int = 0
    caching: bool = True
    method: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    dataset: str = "crossner_literature"
    llm: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    k_shot: int = 5
    number_of_examples: int = 10
    number_of_examples_for_original_dataset: int = 10
    generated: bool = False  # This always needs to be false!
    seed: int = 42
    number_of_examples_per_intent: int = 5
    task: str = "ner"
    number_of_ner: Optional[set] = None
    number_of_examples_per_score: int = 5


# Set of dataclasses meant to specify input and output description for various classes

@dataclass
class NERMolecule:
    start: int
    end: int
    label: str
    text: str


@dataclass
class NERDataPoint:
    text: List
    labels: List
    ners: List[NERMolecule]


@dataclass
class IntentDataPoint:
    text: str
    label: str


@dataclass
class TextSimilarityDataPoint:
    text1: str
    text2: str
    score: float
