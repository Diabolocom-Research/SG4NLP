'''
Runs the text similarity pipeline including generation, testing on generation, and on original/real dataset
'''
from config import *
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from runner import generate_orch, benchmark_orch

use_mlflow: bool = False
dataset: str = "tweet-news"
split: str = "test"
caching: bool = True
k_shot: int = 1
number_of_examples: int = 200
task: str = "text_similarity"
number_of_examples_per_intent: int = 1
use_redis_caching: bool = True
port: int = 6379
number_of_examples_per_score: int = 1
number_of_examples_for_original_dataset: int = 200
seed = 42
number_of_ner = None

big_list_of_llm = ["mistralai/Mixtral-8x7B-Instruct-v0.1",
                   "gpt-4o",
                   "meta-llama/Meta-Llama-3-70B-Instruct",
                   "mistralai/Mixtral-8x22B-Instruct-v0.1",
                   "gpt-4o-mini",
                   "meta-llama/Meta-Llama-3-8B-Instruct"]

methods = big_list_of_llm

# generate dataset

all_generate_dataset_arguments = []

for llm in big_list_of_llm:
    gra = GenerateRunnerArguments(
        caching=caching,
        method="None",
        dataset=dataset,
        llm=llm,
        k_shot=k_shot,
        number_of_examples=number_of_examples,
        number_of_examples_for_original_dataset=number_of_examples_for_original_dataset,
        generated=False,  # This always needs to be false!
        seed=seed,
        number_of_examples_per_intent=number_of_examples_per_intent,
        task=task,
        number_of_ner=number_of_ner,
        number_of_examples_per_score=number_of_examples_per_score
    )

    all_generate_dataset_arguments.append(gra)

results = Parallel(n_jobs=3)(
    delayed(generate_orch)(arguments) for arguments in
    tqdm(all_generate_dataset_arguments))

print(results)

#
# #
all_benchmark_arguments = []
for llm in big_list_of_llm:
    for method in methods:
        all_benchmark_arguments.append(
            BenchmarkRunnerArguments(
            use_mlflow=use_mlflow,
            dataset=dataset,
            method=method,
            split=split,
            caching=caching,
            llm=llm,
            k_shot=k_shot,
            number_of_examples=number_of_examples,
            generated = True,
            seed=seed,
            task=task,
            number_of_examples_per_intent=number_of_examples_per_intent,
            use_redis_caching=use_redis_caching,
            port=port,
            number_of_ner=number_of_ner,
            number_of_examples_per_score=number_of_examples_per_score,
            number_of_examples_for_original_dataset=number_of_examples_for_original_dataset
        ))


print(len(all_benchmark_arguments))

results = Parallel(n_jobs=3)(delayed(benchmark_orch)(arguments) for arguments in
                                 tqdm(all_benchmark_arguments))
print(results)




all_benchmark_arguments = []
for method in methods:
    all_benchmark_arguments.append(
        BenchmarkRunnerArguments(
        use_mlflow=use_mlflow,
        dataset=dataset,
        method=method,
        split=split,
        caching=caching,
        llm=method,
        k_shot=k_shot,
        number_of_examples=number_of_examples,
        generated = False,
        seed=seed,
        task=task,
        number_of_examples_per_intent=number_of_examples_per_intent,
        use_redis_caching=use_redis_caching,
        port=port,
        number_of_ner=number_of_ner,
        number_of_examples_per_score=number_of_examples_per_score,
        number_of_examples_for_original_dataset=number_of_examples_for_original_dataset
    ))


print(len(all_benchmark_arguments))

results = Parallel(n_jobs=3)(delayed(benchmark_orch)(arguments) for arguments in
                                 tqdm(all_benchmark_arguments))
print(results)

# method, llm, generated