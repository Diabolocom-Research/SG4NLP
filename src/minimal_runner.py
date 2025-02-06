'''The orchestrator which does all the heavy lifting by calling various submodules'''
import random
from dataclasses import asdict
from pprint import pprint

import numpy as np
from dotenv import load_dotenv
from nervaluate import Evaluator

from config import *
from methods import get_predictions
from parse_datasets import dataset_parser


def get_dataset(dataset_params: Dataset, generated_dataset_params: Optional[GenerateDataset] = None):
    """Here we will after all the process read the dataset
    dataset_params is always required
    if generated_dataset is also passed, then we assume that the dataset to retrieve is generated
    """

    if generated_dataset_params:
        # here it would mostly be finding the right config based on the arguments and then retrieve the dataset
        raise NotImplementedError
    else:
        params = asdict(dataset_params)
        params['generated'] = False
        params["datafolder"] = DATA_FOLDER
        dataset = dataset_parser.get_dataset(dataset_name=dataset_params.name, **params)
        if dataset_params.number_of_test_examples != -1:
            dataset['test'] = dataset["test"][:dataset_params.number_of_test_examples]
        return dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def benchmark_orch(
        benchmark_params: MinimalBenchmarkArguments,
        dataset_params: Dataset,
        method_params: MethodArguments,
        generated_dataset_params: Optional[GenerateDataset] = None):
    """
    - Pass None, if one does not want to use the generated dataset
    - Get the appropriate dataset by setting the right params and sending it to the function
    - Send the dataset to the prediction function once again by setting the right params
    - Evaluate the prediction using the right params
    - Store all the config, predictions, and results in one place or folder for later retrival
    """

    load_dotenv()

    set_seed(benchmark_params.seed)

    # retrieve dataset
    dataset = get_dataset(dataset_params=dataset_params, generated_dataset_params=generated_dataset_params)

    # retrieve the llm if the methods uses llm
    # if method_params.method == "llm":
    #     llm_adapter = get_llm_adapter(method_params.method_specific_params["llm_config"])

    # method specific arguments
    method_params.method_specific_params['splits'] = benchmark_params.split  # for legacy reason
    method_params.method_specific_params['task'] = benchmark_params.task  # for legacy reason
    preds = get_predictions.predictions(method_name=method_params.method, dataset=dataset,
                                        **method_params.method_specific_params)

    # evaluate the predictions
    evaluator = Evaluator([[i.__dict__ for i in d.ners] for d in
                           dataset[benchmark_params.split]],
                          [[i.__dict__ for i in d.ners] for d in
                           preds],
                          tags=dataset[benchmark_params.split][0].labels)

    results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
    pprint(results)


if __name__ == "__main__":
    dataset_params = Dataset(name="crossner_politics", number_of_test_examples=50)
    generated_dataset_params = None
    llm_config = LLMConfig()
    method_params = MethodArguments()
    benchmark_params = MinimalBenchmarkArguments()

    benchmark_orch(benchmark_params=benchmark_params, dataset_params=dataset_params, method_params=method_params,
                   generated_dataset_params=generated_dataset_params)
