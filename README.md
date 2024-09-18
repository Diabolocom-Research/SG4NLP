# SG4NLP
Synthetic data generation for NLP


### Repository structure
- ```data``` folder consists of:
	- ```generated``` which consist of all generated dataset
	- ```intent_dataset``` consisting of all datasets related to intent recognition
	- ```ner_dataset``` consisting of all datasets related to name entity recognition
	- ```sts_dataset``` consisting of all text similarity dataset
- ```src``` folder consists of all the codes
  - ```generate_dataset``` module consists of all files related to dataset generation
  - ```method``` module consists of all files related to prediction
  - ```parse_dataset``` module consists of all files related to parsing different datasets
- ```mlflow.zip``` unzip it at the same place. Contains all the predictions of different methods and all tasks.


### Running
- To run one specific configuration change the configuration in ```config.py``` file and then use the ```runner.py``` file to call the pipeline.
- To run multiple configuration and the whole pipeline including generation, testing, and over the orginal dataset
  - ```ner_runner.py``` for Name Entity Recognition
  - ```intent_detection_runner.py``` for Intent Detection
  - ```text_similarity_runner.py``` for Text Similarity

### Benchmarking
- All predictions are stored in mlflow folder. It is a zip file and needs to be unzipped.
- Results and analysis can be accessed by ```mlflow_results_analysis.ipynb``` file.

### Generated Dataset
- All generated datasets are available as pickle file in ```data/generated``` folder.
- The name of the file specifies the exact configuration used to create the file
  - ```atis_gpt-4o-mini_1_200.pkl```: 