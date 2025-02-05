'''The orchestrator which does all the heavy lifting by calling various sub modules'''

from src.config import *
from dataclasses import asdict
from src.parse_datasets import dataset_parser

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
        dataset['test'] = dataset["test"][:dataset_params.number_of_test_examples]
        return dataset



def benchmark_orch(dataset_params:Dataset, generated_dataset_params: Optional[GenerateDataset] = None):
    """
    - Pass None, if one does not want to use the generated dataset
    - Get the appropriate dataset by setting the right params and sending it to the function
    - Send the dataset to the prediction function once again by setting the right params
    - Evaluate the prediction using the right params
    - Store all the config, predictions, and results in one place or folder for later retrival
    """

    dataset = get_dataset(dataset_params=dataset_params, generated_dataset_params=generated_dataset_params)


if __name__ == "__main__":
    dataset_params = Dataset(name="crossner_politics", number_of_test_examples=200)
    generated_dataset_params = None
    benchmark_orch(dataset_params=dataset_params, generated_dataset_params=generated_dataset_params)
