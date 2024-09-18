'''The orchestrator which does all the heavy lifting by calling various sub modules'''
import random
import uuid
import redis
import mlflow
import numpy as np
from dotenv import load_dotenv
from nervaluate import Evaluator

from src.config import *
from src.generate_datasets import generate_datasets
from src.methods import get_predictions
from src.parse_datasets import dataset_parser


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def generate_dataset_string(arguments):
    """The function is used to construct a string representative of the argument such that
    the dataset can be retrived in the future for evaluation purposes"""
    llm = arguments.llm.replace('/', '_')
    if arguments.generated == False:
        return arguments.dataset

    if arguments.task == "ner":
        if arguments.seed == 42:
            generated_data_string = f"{arguments.dataset}_{llm}_{arguments.k_shot}_{arguments.number_of_examples}"
        else:
            generated_data_string = f"{arguments.dataset}_{llm}_{arguments.k_shot}_{arguments.number_of_examples}_{arguments.seed}"
    elif arguments.task == "intent":
        if arguments.seed == 42:
            generated_data_string = f"{arguments.dataset}_{llm}_{arguments.number_of_examples_per_intent}_{arguments.number_of_examples}"
        else:
            generated_data_string = f"{arguments.dataset}_{llm}_{arguments.number_of_examples_per_intent}_{arguments.number_of_examples}_{arguments.seed}"
    elif arguments.task == "text_similarity":
        if arguments.seed == 42:
            generated_data_string = f"{arguments.dataset}_{llm}_{arguments.number_of_examples_per_score}_{arguments.number_of_examples}"
        else:
            generated_data_string = f"{arguments.dataset}_{llm}_{arguments.number_of_examples_per_score}_{arguments.number_of_examples}_{arguments.seed}"
    else:
        raise NotImplementedError

    return generated_data_string


def generate_method_string(dataset_string, method):
    return f"{method}_{dataset_string}"


def benchmark_orch(benchmark_runner_arguments: BenchmarkRunnerArguments):
    load_dotenv()

    set_seed(benchmark_runner_arguments.seed)

    if benchmark_runner_arguments.use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_URI)

    if benchmark_runner_arguments.use_redis_caching:
        r = redis.Redis(host='localhost', port=benchmark_runner_arguments.port,
                        decode_responses=True)
    else:
        r = ""

    # if benchmark_runner_arguments.llm in ["gpt-3.5-turbo","gpt-4o", "mistralai/Mixtral-8x7B-Instruct-v0.1"]:
    #     set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    dataset_params = {
        "datafolder": DATA_FOLDER,
        "generated": benchmark_runner_arguments.generated,
        "k_shot": benchmark_runner_arguments.k_shot,
        "llm": benchmark_runner_arguments.llm,
        "number_of_examples": benchmark_runner_arguments.number_of_examples,
        "task": benchmark_runner_arguments.task,
        "number_of_ner": benchmark_runner_arguments.number_of_ner,
        "number_of_examples_per_intent": benchmark_runner_arguments.number_of_examples_per_intent,
        "dataset_string": generate_dataset_string(benchmark_runner_arguments),
        "number_of_examples_for_original_dataset": benchmark_runner_arguments.number_of_examples_for_original_dataset
    }


    dataset = dataset_parser.get_dataset(benchmark_runner_arguments.dataset,
                                         **dataset_params)

    method_params = {
        "splits": benchmark_runner_arguments.split,
        "dataset_name": benchmark_runner_arguments.dataset,
        "caching": benchmark_runner_arguments.caching,
        "method": benchmark_runner_arguments.method,
        "generated": benchmark_runner_arguments.generated,
        "llm": benchmark_runner_arguments.llm,
        "task": benchmark_runner_arguments.task,
        "k_shot": benchmark_runner_arguments.k_shot,
        "number_of_examples_per_intent": benchmark_runner_arguments.number_of_examples_per_intent,
        "number_of_examples": benchmark_runner_arguments.number_of_examples,
        "redis_client": r,
        "use_redis_caching": benchmark_runner_arguments.use_redis_caching,
        "number_of_ner": benchmark_runner_arguments.number_of_ner

    }

    preds = get_predictions.predictions(benchmark_runner_arguments.method, dataset,
                                        **method_params)

    if benchmark_runner_arguments.task == "ner":

        try:
            evaluator = Evaluator([[i.__dict__ for i in d.ners] for d in
                                   dataset[benchmark_runner_arguments.split]],
                                  [[i.__dict__ for i in d.ners] for d in
                                   preds[benchmark_runner_arguments.split]],
                                  tags=dataset["test"][0].labels)

            results, results_per_tag, result_indices, result_indices_by_tag = evaluator.evaluate()
        except ValueError:
            print(benchmark_runner_arguments)
            raise IOError

        if benchmark_runner_arguments.use_mlflow:
            with mlflow.start_run():
                mlflow.set_tag("mlflow.runName",
                               f"{benchmark_runner_arguments.method}__{str(uuid.uuid4())[:6]}")
                mlflow.log_params(benchmark_runner_arguments.__dict__)
                flatten_results = {}

                for result_type, result_value in results.items():
                    for key, value in result_value.items():
                        flatten_results[f"{result_type}_{key}"] = value

                mlflow.log_metrics(flatten_results)


    elif benchmark_runner_arguments.task == "intent":
        encoder = LabelBinarizer()
        encode_label = encoder.fit(dataset["extra"]["labels"])
        y_preds = encode_label.transform(preds[benchmark_runner_arguments.split])
        y_true = encode_label.transform(
            [t.label for t in dataset[benchmark_runner_arguments.split]])

        cr = classification_report(
            y_true=y_true,
            y_pred=y_preds,
            output_dict=True
        )

        if benchmark_runner_arguments.use_mlflow:
            with mlflow.start_run():
                mlflow.set_tag("mlflow.runName",
                               f"{benchmark_runner_arguments.method}__{str(uuid.uuid4())[:6]}")
                mlflow.log_params(benchmark_runner_arguments.__dict__)
                # mlflow.log_metric("accuracy", cr.pop("accuracy"))
                for class_or_avg, metrics_dict in cr.items():
                    for metric, value in metrics_dict.items():
                        mlflow.log_metric(class_or_avg + '_' + metric, value)

        results = cr

    elif benchmark_runner_arguments.task == "text_similarity":
        y_pred = preds[benchmark_runner_arguments.split]
        y_true = [t.score for t in dataset[benchmark_runner_arguments.split]]
        pr = pearsonr(y_pred, y_true)[0]
        sr = spearmanr(y_pred, y_true)[0]
        e = mse(y_pred, y_true)
        print(pr, sr, e)
        results = [pr, sr, e]

        if benchmark_runner_arguments.use_mlflow:
            with mlflow.start_run():
                mlflow.set_tag("mlflow.runName",
                               f"{benchmark_runner_arguments.method}__{str(uuid.uuid4())[:6]}")
                mlflow.log_params(benchmark_runner_arguments.__dict__)
                # mlflow.log_metric("accuracy", cr.pop("accuracy"))
                mlflow.log_metric("pr", pr)
                mlflow.log_metric("sr", sr)
                mlflow.log_metric("e", e)

    return results

    # print(dataset["test"][0].labels)


def generate_orch(generate_runner_arguments: GenerateRunnerArguments):
    """Generates the dataset based on the arguments provided by GenerateRunnerArguments"""
    load_dotenv()
    set_seed(generate_runner_arguments.seed)
    dataset_params = {
        "datafolder": DATA_FOLDER,
        "generated": generate_runner_arguments.generated,
        "number_of_ner": generate_runner_arguments.number_of_ner,
        "number_of_examples_for_original_dataset": generate_runner_arguments.number_of_examples_for_original_dataset
    }

    dataset = dataset_parser.get_dataset(generate_runner_arguments.dataset,
                                         **dataset_params)

    generate_runner_arguments.generated = True

    generator_params = {
        'caching': generate_runner_arguments.caching,
        'k_shot': generate_runner_arguments.k_shot,
        'number_of_examples': generate_runner_arguments.number_of_examples,
        'number_of_examples_per_intent': generate_runner_arguments.number_of_examples_per_intent,
        "task": generate_runner_arguments.task,
        "number_of_ner": generate_runner_arguments.number_of_ner,
        "number_of_examples_per_score": generate_runner_arguments.number_of_examples_per_score,
        "dataset_string": generate_dataset_string(generate_runner_arguments)
    }
    generate_datasets.router(llm=generate_runner_arguments.llm, dataset=dataset,
                             **generator_params)

    return "done"


if __name__ == "__main__":
    methods = ["gpt-4o", "gpt-3.5-turbo", "meta-llama/Meta-Llama-3-70B-Instruct",
               "mistralai/Mixtral-8x7B-Instruct-v0.1",
               "mistralai/Mixtral-8x22B-Instruct-v0.1", "gpt-4o-mini"]

    print(benchmark_orch(benchmark_runner_arguments=BenchmarkRunnerArguments()))
