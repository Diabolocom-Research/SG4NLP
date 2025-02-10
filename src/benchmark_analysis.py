import os
import re
import mlflow
import numpy as np
import pandas as pd
from config import *
from pprint import pprint
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from mlflow.tracking import MlflowClient

project_dir = os.getcwd()
parent_dir = os.path.dirname(project_dir)
mlflow_dir = os.path.join(parent_dir, "mlflow_v2")
mlflow.set_tracking_uri("file://" + mlflow_dir)

experiment_id = "0"
runs_df: pd.DataFrame = mlflow.search_runs(experiment_ids=[experiment_id])


all_llms = ["meta-llama/llama-3.1-70b-instruct", "mistralai/mixtral-8x22b-instruct", "gpt-4o-mini", "anthropic/claude-3-haiku", "google/gemini-2.0-flash-001"]
all_datasets = ['crossner_politics', 'crossner_literature', 'crossner_science']
# current_llm = "meta-llama/llama-3.1-70b-instruct"
# current_dataset = "crossner_politics"
current_metric = "metrics.strict_f1"

def filter_by_dataset_name(df: pd.DataFrame, target_name: str = "crossner_politics") -> pd.DataFrame:

    def apply_literal_eval(dataset_str:str):
        if dataset_str and dataset_str != None and dataset_str != "":
            return eval(dataset_str)

    # Create a boolean mask where the extracted name equals target_name.
    # mask = df["params.dataset"].apply(lambda s: check_for_empty_str(s) != False)
    # df = df[mask]
    mask = df["params.dataset_params"].apply(lambda s: apply_literal_eval(s)['name'] == target_name)
    return df[mask]


def filter_by_method_name(df: pd.DataFrame, target_name: str = "meta-llama/llama-3.1-70b-instruct") -> pd.DataFrame:

    def apply_literal_eval(dataset_str:str):
        if dataset_str and dataset_str != None and dataset_str != "":
            return eval(dataset_str)

    # Create a boolean mask where the extracted name equals target_name.
    # mask = df["params.dataset"].apply(lambda s: check_for_empty_str(s) != False)
    # df = df[mask]
    mask = df["params.method_params"].apply(lambda s: apply_literal_eval(s)['method_specific_params']['llm_config'].model_name == target_name)
    return df[mask]


def filter_by_llm_generated(df: pd.DataFrame, target_name: str = "meta-llama/llama-3.1-70b-instruct") -> pd.DataFrame:
    '''Only works where generated_dataset_params'''

    def apply_literal_eval(dataset_str:str):
        if dataset_str and dataset_str != None and dataset_str != "":
            a = eval(dataset_str)
            return a

    # Create a boolean mask where the extracted name equals target_name.
    # mask = df["params.dataset"].apply(lambda s: check_for_empty_str(s) != False)
    # df = df[mask]
    mask = df["params.generated_dataset_params"].apply(lambda s: apply_literal_eval(s)['llm_for_generation'] == target_name)
    return df[mask]


for current_dataset in all_datasets:
    for curent_llm1 in all_llms:
        print(f"-------{curent_llm1}---------{current_dataset}---------------")
        for llm in all_llms:
            # We will first get the results on real dataset
            # Then we will get results on generated dataset

            # remove rows related to generated dataset
            mask_not_gen = runs_df["params.generated_dataset_params"].isna()
            temp = runs_df[mask_not_gen]

            # find rows related to current_dataset
            temp = filter_by_dataset_name(df=temp, target_name=current_dataset)

            # find the row related to the llm of the loop
            temp = filter_by_method_name(df=temp, target_name=llm)

            # get results and also assert len
            assert len(temp) == 1
            results_on_real_dataset = temp[current_metric].item()

            # now do the same with generated
            # remove all the rows which have not generated dataset
            mask_gen = runs_df["params.generated_dataset_params"].notna()
            temp = runs_df[mask_gen]

            # find rows related to current_dataset
            temp = filter_by_dataset_name(df=temp, target_name=current_dataset)

            # specifically find all the rows related to current llm
            temp = filter_by_llm_generated(temp, curent_llm1)

            # find the row related to the llm of the loop
            temp = filter_by_method_name(df=temp, target_name=llm)

            assert len(temp) == 1
            results_on_gen_dataset = temp[current_metric].item()

            print(llm, round(results_on_real_dataset, 3), round(results_on_gen_dataset, 3))


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
        current_llm = llm
        current_dataset = dataset
        # bias facot of current_llm on current_dataset is
        _a = get_normalized_performance_v2(current_llm=current_llm, current_dataset=current_dataset, current_method=current_llm)
        _b = np.mean([get_normalized_performance_v2(current_llm=i, current_dataset=current_dataset, current_method=current_llm) for i in all_llms if i != current_llm])




