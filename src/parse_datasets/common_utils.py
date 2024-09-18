# Essentially a copy from - gliner evaluate.py

import os
import glob
import json


def flatten_ner(tokenized_text, ner_list):
    flatten_ner = []
    for ner in ner_list:
        i, j, label = ner
        start = len(" ".join(tokenized_text[:i]))
        end = len(" ".join(tokenized_text[:j + 1]))
        temp = {
            "start": start + 1,
            "end": end,
            "label": label,
            "text": " ".join(tokenized_text)[start:end]
        }
        flatten_ner.append(temp)

    return flatten_ner


def open_content(path):
    paths = glob.glob(os.path.join(path, "*.json"))
    train, dev, test, labels = None, None, None, None
    for p in paths:
        if "train" in p:
            with open(p, "r") as f:
                train = json.load(f)
        elif "dev" in p:
            with open(p, "r") as f:
                dev = json.load(f)
        elif "test" in p:
            with open(p, "r") as f:
                test = json.load(f)
        elif "labels" in p:
            with open(p, "r") as f:
                labels = json.load(f)
    return train, dev, test, labels

def process(data):
    words = data['sentence'].split()
    entities = []  # List of entities (start, end, type)

    for entity in data['entities']:
        start_char, end_char = entity['pos']

        # Initialize variables to keep track of word positions
        start_word = None
        end_word = None

        # Iterate through words and find the word positions
        char_count = 0
        for i, word in enumerate(words):
            word_length = len(word)
            if char_count == start_char:
                start_word = i
            if char_count + word_length == end_char:
                end_word = i
                break
            char_count += word_length + 1  # Add 1 for the space

        # Append the word positions to the list
        entities.append((start_word, end_word, entity['type'].lower()))

    # Create a list of word positions for each entity
    sample = {
        "tokenized_text": words,
        "ner": entities
    }

    return sample


# create dataset
def create_dataset(path):
    train, dev, test, labels = open_content(path)
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for data in train:
        train_dataset.append(process(data))
    for data in dev:
        dev_dataset.append(process(data))
    for data in test:
        test_dataset.append(process(data))
    labels = [label.lower() for label in labels]
    return train_dataset, dev_dataset, test_dataset, labels