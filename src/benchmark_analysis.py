import os

import mlflow
import numpy as np
import pandas as pd
from config import *

project_dir = os.getcwd()
parent_dir = os.path.dirname(project_dir)
mlflow_dir = os.path.join(parent_dir, "mlflow_v2")
mlflow.set_tracking_uri("file://" + mlflow_dir)

experiment_id = "0"
runs_df: pd.DataFrame = mlflow.search_runs(experiment_ids=[experiment_id])

# all_llms = ["meta-llama/llama-3.1-70b-instruct", "mistralai/mixtral-8x22b-instruct", "gpt-4o-mini",
#             "anthropic/claude-3-haiku", "google/gemini-2.0-flash-001"]
# all_llms = ["meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct", "meta-llama/llama-3.1-405b-instruct"]
# all_llms = ["anthropic/claude-3-sonnet", "anthropic/claude-3-haiku"]

all_llms = ["gpt-4o","gpt-4o-mini"]


all_datasets = ['crossner_politics', 'crossner_literature', 'crossner_science']
# all_datasets = ['crossner_politics']

# current_llm = "meta-llama/llama-3.1-70b-instruct"
# current_dataset = "crossner_politics"
current_metric = "metrics.exact_f1"
from scipy import stats

def filter_by_dataset_name(df: pd.DataFrame, target_name: str = "crossner_politics") -> pd.DataFrame:
    def apply_literal_eval(dataset_str: str):
        if dataset_str and dataset_str != None and dataset_str != "":
            return eval(dataset_str)

    # Create a boolean mask where the extracted name equals target_name.
    # mask = df["params.dataset"].apply(lambda s: check_for_empty_str(s) != False)
    # df = df[mask]
    mask = df["params.dataset_params"].apply(lambda s: apply_literal_eval(s)['name'] == target_name)
    return df[mask]


def filter_by_method_name(df: pd.DataFrame, target_name: str = "meta-llama/llama-3.1-70b-instruct") -> pd.DataFrame:
    def apply_literal_eval(dataset_str: str):
        if dataset_str and dataset_str != None and dataset_str != "":
            return eval(dataset_str)

    # Create a boolean mask where the extracted name equals target_name.
    # mask = df["params.dataset"].apply(lambda s: check_for_empty_str(s) != False)
    # df = df[mask]
    mask = df["params.method_params"].apply(
        lambda s: apply_literal_eval(s)['method_specific_params']['llm_config'].model_name == target_name)
    return df[mask]


def filter_by_llm_generated(df: pd.DataFrame, target_name: str = "meta-llama/llama-3.1-70b-instruct") -> pd.DataFrame:
    '''Only works where generated_dataset_params'''

    def apply_literal_eval(dataset_str: str):
        if dataset_str and dataset_str != None and dataset_str != "":
            a = eval(dataset_str)
            return a

    # Create a boolean mask where the extracted name equals target_name.
    # mask = df["params.dataset"].apply(lambda s: check_for_empty_str(s) != False)
    # df = df[mask]
    mask = df["params.generated_dataset_params"].apply(
        lambda s: apply_literal_eval(s)['llm_for_generation'] == target_name)
    return df[mask]


def return_performace_on_real_and_gen_dataset(runs_df, method, llm_for_gen, dataset, current_metric):
    # Step 1 - Find performance on real dataset
    mask_not_gen = runs_df["params.generated_dataset_params"].isna()
    temp = runs_df[mask_not_gen]

    # find rows related to current_dataset
    temp = filter_by_dataset_name(df=temp, target_name=dataset)

    # find the row related to the llm of the loop
    temp = filter_by_method_name(df=temp, target_name=method)

    # get results and also assert len
    assert len(temp) == 1
    results_on_real_dataset = temp[current_metric].item()

    # Step 2 - Find performance on the generated dataset
    mask_gen = runs_df["params.generated_dataset_params"].notna()
    temp = runs_df[mask_gen]

    # find rows related to current_dataset
    temp = filter_by_dataset_name(df=temp, target_name=dataset)

    # specifically find all the rows related to current llm
    temp = filter_by_llm_generated(temp, llm_for_gen)

    # find the row related to the llm of the loop
    temp = filter_by_method_name(df=temp, target_name=method)

    assert len(temp) == 1
    results_on_gen_dataset = temp[current_metric].item()

    return results_on_real_dataset, results_on_gen_dataset


for current_dataset in all_datasets:
    for curent_llm1 in all_llms:
        print(f"-------{curent_llm1}---------{current_dataset}---------------")
        performances = []
        for llm in all_llms:
            results_on_real_dataset, results_on_gen_dataset = return_performace_on_real_and_gen_dataset(runs_df,
                                                                                                        method=llm,
                                                                                                        llm_for_gen=curent_llm1,
                                                                                                        dataset=current_dataset,
                                                                                                        current_metric=current_metric)

            # print(llm, round(results_on_real_dataset, 3), round(results_on_gen_dataset, 3))
            performances.append([results_on_real_dataset, results_on_gen_dataset])
        mpsd = np.mean([abs(real-gen) for real, gen in performances])
        res = stats.spearmanr([real for real, _ in performances], [gen for _, gen in performances])
        print(round(mpsd,2), round(res.statistic,3))


def get_normalized_performance_v2(current_llm, current_dataset, current_method):
    '''
        current_llm and current_dataset would be used for figuring out the dataset generated with current llm D^i
        current method would be used for figuring out which to use
    '''
    mask_gen = runs_df["params.generated_dataset_params"].notna()  # get all generated dataset
    temp = runs_df[mask_gen]

    temp = filter_by_dataset_name(df=temp, target_name=current_dataset)  # find rows related to current_dataset
    temp = filter_by_llm_generated(temp, current_llm)  # specifically find all the rows related to current llm
    temp = filter_by_method_name(df=temp, target_name=current_method)  # find the row related to the llm of the loop

    assert len(temp) == 1
    result_of_current_llm_on_its_own_dataset = temp[current_metric].item()

    all_r = []
    # Step 2a - performance of all llm on the dataset generated by the current_llm on dataset current_dataset
    for llm in all_llms:
        if llm != current_llm:
            mask_gen = runs_df["params.generated_dataset_params"].notna()  # get all generated dataset
            temp = runs_df[mask_gen]

            temp = filter_by_dataset_name(df=temp, target_name=current_dataset)  # find rows related to current_dataset
            temp = filter_by_llm_generated(temp, current_llm)  # specifically find all the rows related to current llm
            temp = filter_by_method_name(df=temp, target_name=llm)  # find the row related to the llm of the loop

            assert len(temp) == 1
            r = temp[current_metric].item()
            all_r.append(r)

    normalized_performance = result_of_current_llm_on_its_own_dataset - np.mean(all_r)

    return normalized_performance


for llm in all_llms:
    for dataset in all_datasets:
        print(f"------{llm}-------{dataset}-------")
        current_llm = llm
        current_dataset = dataset
        # bias facot of current_llm on current_dataset is
        _a = get_normalized_performance_v2(current_llm=current_llm, current_dataset=current_dataset,
                                           current_method=current_llm)
        _b = np.mean(
            [get_normalized_performance_v2(current_llm=i, current_dataset=current_dataset, current_method=current_llm)
             for i in all_llms if i != current_llm])

        print(round(_a - _b, 3))



def average_performance(runs_df, method, dataset):
    # Step 1 - Find performance on real dataset
    mask_not_gen = runs_df["params.generated_dataset_params"].isna()
    temp = runs_df[mask_not_gen]

    # find rows related to current_dataset
    temp = filter_by_dataset_name(df=temp, target_name=dataset)

    # find the row related to the llm of the loop
    temp = filter_by_method_name(df=temp, target_name=method)

    # get results and also assert len
    assert len(temp) == 1
    results_on_real_dataset = temp[current_metric].item()

    # Step 2 - Find performance on the generated dataset


    # specifically find all the rows related to current llm
    r = []
    for llm_for_gen in all_llms:
        mask_gen = runs_df["params.generated_dataset_params"].notna()
        temp = runs_df[mask_gen]

        # find rows related to current_dataset
        temp = filter_by_dataset_name(df=temp, target_name=dataset)
        temp = filter_by_llm_generated(temp, llm_for_gen)

        # find the row related to the llm of the loop
        temp = filter_by_method_name(df=temp, target_name=method)
        assert len(temp) == 1
        r.append(temp[current_metric].item())

    return results_on_real_dataset, np.mean(r)




for current_dataset in all_datasets:
        print(f"-------avg performance---------{current_dataset}---------------")
        performances = []
        for llm in all_llms:
            results_on_real_dataset, results_on_gen_dataset = average_performance(runs_df=runs_df,
                                                                                  method=llm,
                                                                                  dataset=current_dataset,)

            # print(llm, round(results_on_real_dataset, 3), round(results_on_gen_dataset, 3))
            performances.append([results_on_real_dataset, results_on_gen_dataset])
        mpsd = np.mean([abs(real-gen) for real, gen in performances])
        res = stats.spearmanr([real for real, _ in performances], [gen for _, gen in performances])
        print(round(mpsd,2), round(res.statistic,3))