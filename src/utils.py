'''Set of utility function. Primarily redis caching layer'''
import pickle
import time
from pathlib import Path
from typing import List, Any, Optional

from config import LLMConfig, Dataset, GenerateDataset
from llms import llm_creation


def caching_layer(model_name, model_temperature, message, llm, redis_cli,
                  use_redis_caching):
    sleep_time = 1
    if model_temperature != 0.0 or use_redis_caching is False:
        time.sleep(sleep_time)
        output = llm.invoke(message)
        if type(output) != str:
            output = output.content

        return output

    composite_key = str(model_name) + str(model_temperature) + str(message)
    cached_response = redis_cli.get(composite_key)
    if cached_response:
        return cached_response
    else:
        time.sleep(sleep_time)
        output = llm.invoke(message)
        if type(output) != str:
            output = output.content
        redis_cli.set(composite_key, output)
        return output


def get_llm_adapter(llm_config: LLMConfig):
    # Create LLM instance using the factory.
    llm_factory = llm_creation.LLMFactory()
    llm_adapter = llm_factory.create_llm(llm_config)
    return llm_adapter


def llm_config_generator(llm_name="gpt-4o", temperature=0.0):
    """It is a helper class which creates llm config based on the name"""

    if llm_name.lower() in ["gpt-4o", "gpt-4o-mini"]:
        project_string = "openai" + llm_name
        llm_config = LLMConfig(llm_server="openai", model_name=llm_name, caching=True, redis_port=6379,
                               redis_host="localhost", project_string=project_string, temperature=temperature)

    elif llm_name.lower() in ["llama-3.1-70b-q4"]:
        project_string = "diabolocom" + llm_name
        llm_config = LLMConfig(llm_server="diabolocom", model_name=llm_name, caching=True, redis_port=6379,
                               redis_host="localhost", project_string=project_string, temperature=temperature)

    else:
        raise NotImplementedError

    return llm_config


def store_dataclasses(instances: dict, file_path: Path) -> None:
    """
    Stores a dictionary of dataclass instances to a file using pickle.

    Args:
        instances (dict): A dictionary of dataclass instances keyed by a name.
        file_path (str): Path to the file where the data should be stored.
    """
    with open(file_path, "wb") as f:
        pickle.dump(instances, f)
    print(f"Dataclass instances have been stored to {file_path}")


def load_dataclasses(file_path: str) -> dict:
    """
    Loads a dictionary of dataclass instances from a file using pickle.

    Args:
        file_path (str): Path to the file from which to load the dataclass instances.

    Returns:
        dict: The dictionary of dataclass instances.
    """
    with open(file_path, "rb") as f:
        instances = pickle.load(f)
    print(f"Dataclass instances have been loaded from {file_path}")
    return instances


def retrive_generated_dataset(dataset_params: Dataset,
                              generated_dataset_params: GenerateDataset,
                              llm_config: LLMConfig,
                              path: Path) -> Optional[List[Any]]:
    """
    Iterates over all pickle files in the given directory and returns a list of
    generated datasets from the files whose stored parameters match the provided ones.

    If multiple files match the criteria, the list is sorted in descending order
    by the file's modification time (i.e. index 0 contains the latest file).

    Args:
        dataset_params (Dataset): The dataset parameters to match.
        generated_dataset_params (GenerateDataset): The generated dataset parameters to match.
        llm_config (LLMConfig): The LLM configuration to match.
        path (Path): The directory where pickle files are stored.

    Returns:
        List[Any]: A list of generated datasets (from the key "generated_dataset")
                   sorted with the latest file first. Returns an empty list if no matches are found.
    """
    matching_items = []

    # Iterate over all pickle files in the specified directory
    for pkl_file in path.glob("*.pkl"):
        try:
            with pkl_file.open("rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print("Error loading file %s: %s", pkl_file, e)
            continue

        # Check if the stored parameters match the provided ones.
        if (data.get("dataset_params") == dataset_params and
                data.get("generate_dataset_params") == generated_dataset_params and
                data.get("llm_config") == llm_config):
            # Get the modification time of the file
            mod_time = pkl_file.stat().st_mtime
            # Append a tuple of (modification_time, generated_dataset)
            matching_items.append((mod_time, data.get("generated_dataset")))

    # Sort by modification time in descending order (latest file first)

    if len(matching_items) == 0:
        return None
    matching_items.sort(key=lambda x: x[0], reverse=True)

    # Extract and return only the generated_dataset objects
    return [item[1] for item in matching_items]
