# SG4NLP
Synthetic data generation for NLP



### Running
- To run one specific configuration change the configuration in ```config.py``` file and then use the ```runner.py``` file to call the pipeline.
- To run multiple configuration and the whole pipeline including generation, testing, and over the orginal dataset
  - ```ner_runner.py``` for name entity recognition
  - ```intent_detection_runner.py``` for intent detection
  - ```text_similarity_runner.py``` for text similarity

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