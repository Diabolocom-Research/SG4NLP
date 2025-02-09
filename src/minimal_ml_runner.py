from config import *
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from minimal_runner import benchmark_orch, generate_dataset_orch

llms = [("openrouter", "meta-llama/llama-3.1-70b-instruct"),
                          ("openrouter", "mistralai/mixtral-8x22b-instruct"), ("openai", "gpt-4o-mini"),
                          ("openrouter", "anthropic/claude-3-haiku")]  # of the form llm_server, llm_model
all_datasets: List[str] = ['crossner_politics', 'crossner_science']

# Benchmark all LLMS on individual dataset
all_benchmark_args = []
for llm_server, model in llms:
    for dataset in all_datasets:
        print(llm_server, model, dataset)
        dataset_params = Dataset(name=dataset, number_of_test_examples=25)
        generated_dataset_params = None
        llm_config = LLMConfig(
            llm_server=llm_server,
            model_name=model,
            caching=True,
            redis_port=6379,
            redis_host="localhost",
            project_string=llm_server + model,
            temperature=0.0
        )

        method_params = MethodArguments(method="llm", method_specific_params={"llm_config": llm_config})
        benchmark_params = MinimalBenchmarkArguments()

        benchmark_arg = {
            "dataset_params": dataset_params,
            "generated_dataset_params": generated_dataset_params,
            # "llm_config": llm_config,
            "method_params": method_params,
            "benchmark_params": benchmark_params
        }

        all_benchmark_args.append(benchmark_arg)


        # bo = benchmark_orch(benchmark_params=benchmark_params, dataset_params=dataset_params,
        #                     method_params=method_params,
        #                     generated_dataset_params=generated_dataset_params)



results = Parallel(n_jobs=5)(
    delayed(benchmark_orch)(**arguments)
    for arguments in tqdm(all_benchmark_args)
)



all_generate_args = []
# Generate Dataset
for llm_server, model in llms:
    for dataset in all_datasets:
        print(llm_server, model, dataset)
        dataset_params = Dataset(name=dataset, number_of_test_examples=25)
        generated_dataset_params = GenerateDataset(dataset=dataset_params, llm_for_generation=model,
                                                   k_shot=5)
        arguments = {
            "dataset_params": dataset_params,
            "generate_dataset_params": generated_dataset_params
        }

        all_generate_args.append(arguments)

results = Parallel(n_jobs=5)(
    delayed(generate_dataset_orch)(**arguments)
    for arguments in tqdm(all_generate_args)
)

# id = generate_dataset_orch(dataset_params=dataset_params, generate_dataset_params=generated_dataset_params)

# Now benchmarking on the generated dataset
# TODO: CHECK IF MODEL_GEN and LLM_SERVER_GEN ARE CORRECTLY ASSIGNED
all_benchmark_args = []
for llm_server, model in llms:  # loop for the method
    for dataset in all_datasets:  # loops for generating dataset - 1
        for llm_server_gen, model_gen in llms[::-1]:  # loops for generating dataset - 2
            # llm config is only for the benchmarking. For generated we would just need name
            print(llm_server, model, dataset, llm_server_gen, model_gen)
            dataset_params = Dataset(name=dataset, number_of_test_examples=25)
            generated_dataset_params = GenerateDataset(dataset=dataset_params, llm_for_generation=model_gen,
                                                       k_shot=5)
            llm_config = LLMConfig(
                llm_server=llm_server,
                model_name=model,
                caching=True,
                redis_port=6379,
                redis_host="localhost",
                project_string=llm_server + model,
                temperature=0.0
            )

            method_params = MethodArguments(method="llm", method_specific_params={"llm_config": llm_config})
            benchmark_params = MinimalBenchmarkArguments()

            benchmark_arg = {
                "dataset_params": dataset_params,
                "generated_dataset_params": generated_dataset_params,
                # "llm_config": llm_config,
                "method_params": method_params,
                "benchmark_params": benchmark_params
            }

            all_benchmark_args.append(benchmark_arg)


            # bo = benchmark_orch(benchmark_params=benchmark_params, dataset_params=dataset_params,
            #                     method_params=method_params,
            #                     generated_dataset_params=generated_dataset_params)

results = Parallel(n_jobs=5)(
    delayed(benchmark_orch)(**arguments)
    for arguments in tqdm(all_benchmark_args)
)