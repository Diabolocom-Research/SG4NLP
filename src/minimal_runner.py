'''The orchestrator which does all the heavy lifting by calling various submodules'''
import os
import random
from dataclasses import asdict
from pprint import pprint

import mlflow
import numpy as np
import shortuuid
from dotenv import load_dotenv
from nervaluate import Evaluator

from config import *
from generate_datasets import llm_based_ner
from methods import get_predictions
from parse_datasets import dataset_parser
from utils import llm_config_generator, store_dataclasses, retrive_generated_dataset


def get_dataset(dataset_params: Dataset, generated_dataset_params: Optional[GenerateDataset] = None):
    """Here we will after all the process read the dataset
    dataset_params is always required
    if generated_dataset is also passed, then we assume that the dataset to retrieve is generated
    """

    if generated_dataset_params:
        # here it would mostly be finding the right config based on the arguments and then retrieve the dataset
        llm_config = llm_config_generator(llm_name=generated_dataset_params.llm_for_generation, temperature=0.0)
        dataset = retrive_generated_dataset(dataset_params, generated_dataset_params, llm_config,
                                            path=Path("../data/generated/v2"))[0]
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

    if benchmark_params.use_mlflow:
        project_dir = os.getcwd()
        parent_dir = os.path.dirname(project_dir)
        mlflow_dir = os.path.join(parent_dir, "mlflow_v2")
        mlflow.set_tracking_uri("file://" + mlflow_dir)
        print("MLflow tracking URI set to:", mlflow.get_tracking_uri())
        uuid = shortuuid.uuid()
        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName",
                           f"{benchmark_params.task}__{uuid[:6]}")
            mlflow.log_params(benchmark_params.__dict__)
            mlflow.log_params(dataset_params.__dict__)
            mlflow.log_params(method_params.__dict__)
            if generated_dataset_params:
                mlflow.log_params(generated_dataset_params.__dict__)
            flatten_results = {}

            for result_type, result_value in results.items():
                for key, value in result_value.items():
                    flatten_results[f"{result_type}_{key}"] = value

            mlflow.log_metrics(flatten_results)


def generate_dataset_orch(dataset_params: Dataset,
                          generate_dataset_params: GenerateDataset):
    dataset = get_dataset(dataset_params, None)
    llm_config = llm_config_generator(llm_name=generate_dataset_params.llm_for_generation, temperature=0.0)
    print("warning: do check temperature before generating")
    retrived_dataset = retrive_generated_dataset(dataset_params=dataset_params, generated_dataset_params=generated_dataset_params,
                              llm_config=llm_config, path=Path("../data/generated/v2"))
    if retrived_dataset:
        return "Generated Dataset with the same exact param already exist!"

    generated_dataset = llm_based_ner.generate_dataset(dataset=dataset,
                                                       generated_dataset_params=generated_dataset_params,
                                                       llm_config=llm_config)
    # save the generated dataset - save dataset params, generated dataset params, llm config, and the dataset itself

    dataset_dict = {
        "generated_dataset": generated_dataset,
        "dataset_params": dataset_params,
        "generate_dataset_params": generate_dataset_params,
        "llm_config": llm_config
    }
    id = shortuuid.uuid()
    store_dataclasses(dataset_dict, Path(f"../data/generated/v2/{id}.pkl"))

    return id


if __name__ == "__main__":
    dataset_params = Dataset(name="crossner_politics", number_of_test_examples=50)
    generated_dataset_params = GenerateDataset(dataset=dataset_params, llm_for_generation="llama-3.1-70b-q4", k_shot=5)
    llm_config = LLMConfig()
    method_params = MethodArguments()
    benchmark_params = MinimalBenchmarkArguments()
    benchmark_orch(benchmark_params=benchmark_params, dataset_params=dataset_params, method_params=method_params,
                   generated_dataset_params=generated_dataset_params)

    # generated_dataset_params = GenerateDataset(dataset=dataset_params, llm_for_generation="llama-3.1-70b-q4", k_shot=5)
    # id = generate_dataset_orch(dataset_params=dataset_params, generate_dataset_params=generated_dataset_params)
    # print(id)
