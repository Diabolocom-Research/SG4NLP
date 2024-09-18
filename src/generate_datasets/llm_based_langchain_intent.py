import os
import re
import pickle
import random
from pathlib import Path
from tqdm.auto import tqdm
from src.config import IntentDataPoint
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Anyscale
from src.config import gpt_llm, claude_llm, anyscale_llm

def clean_output(text):
    pattern = r'generated_text>(.*?)</generated_text'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # Extract and print the text if the pattern is found
    if match:
        extracted_text = match.group(1).strip()
        return extracted_text
    else:
        return "No text found between <generated_text> tags."


def find_representative_examples(candidate_labels, dataset, k):
    example = []
    for cl in candidate_labels:
        c = 0
        for t in dataset:
            if t.label == cl:
                example.append(t)
                c = c + 1
                if c == k:
                    break

    final_string = ""
    for e in example:
        final_string = final_string + "Intent:" + e.label + "\n"
        final_string = final_string + "Text:" + e.text + "\n"
        final_string = final_string + "\n"

    return final_string


def generate_examples(example_string, number_of_examples, llm, all_intents):
    generation_user_prompt = '''You are an AI assistant tasked with generating different short text sample for evaluating intent recognition systems. Your goal is to create a creative text that is representative of the intent provided by the user. Some examples are

{example_string}
Follow these instructions to generate the text:

1. Generate a text with the intent provided by the user.
2. The size of text shoule be similar to the examples provided above
3. Be creative and keep the text natural, following the style and tone of the examples provided above.
4. Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).
5. Make absolutely sure the generated text has the same intent as provided by the user


After generating the text, please provide your output in the following format:

</generated_text>
[Your generated text here]
</generated_text>


Remember, the goal is to create realistic-looking sample that could be used to test intent recognition systems. Make sure your text sounds natural and incorporates the intent provided by the user.'''

    generated_dataset = []
    counter = 0
    pbar = tqdm(total=number_of_examples)
    while len(generated_dataset) < number_of_examples:
        if counter > number_of_examples + 200:
            break
        counter += 1
        # find the intent
        intent = random.sample(all_intents, 1)[0]
        messages = [
            (
                "system",
                generation_user_prompt.format(example_string=example_string),
            ),
            ("human", intent),
        ]

        output = llm.invoke(messages)
        # print(messages)
        # print(output)
        if type(output) != str:
            output = output.content

        print(output)

        text = clean_output(output)
        print(text)
        if text != "No text found between <generated_text> tags.":
            generated_dataset.append(
                IntentDataPoint(label=intent, text=text))
            pbar.update(1)

    return generated_dataset

def generate_dataset(dataset, llm, **kwargs):
    # print(
    #     "you have not fixed generated string thingy and as well as the folder location. Fix it first!")
    # raise IOError


    dataset_theme = dataset["extra"]["theme"]
    dataset_name = dataset["extra"]["name"]
    dataset_labels = dataset["extra"]["labels"]
    k = kwargs["number_of_examples_per_intent"]
    datasaving_string = kwargs["dataset_string"]


    temperature = 0.99
    if llm in gpt_llm:
        model = ChatOpenAI(
            model=llm,
            temperature=temperature,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    elif llm in anyscale_llm:
        model = Anyscale(model_name=llm, temperature=temperature)
    elif llm in claude_llm:
        model = ChatAnthropic(
            model=llm,
            temperature=temperature,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    else:
        raise NotImplementedError

    # Step 1a: Generate Labels
    # Path("../generation/").mkdir(parents=True, exist_ok=True)

    # Step 2a: Select examples
    example_string = find_representative_examples(dataset_labels, dataset["train"], k)
    fname = f"../data/generated/{datasaving_string}.pkl"


    # Step 3: Generate examples
    if kwargs["caching"]:
        # fname = f"../generation/{dataset_name}_{llm.replace('/', '_')}_{kwargs['number_of_examples_per_intent']}_{kwargs['number_of_examples']}.pkl"
        if os.path.isfile(fname):
            all_examples = pickle.load(open(fname, "rb"))
        else:
            all_examples = generate_examples(example_string=example_string,
                                             number_of_examples=kwargs[
                                                 'number_of_examples'], llm=model,
                                             all_intents=dataset_labels)
            pickle.dump(all_examples, open(fname, "wb"))
    else:
        all_examples = generate_examples(example_string=example_string,
                                         number_of_examples=kwargs[
                                             'number_of_examples'], llm=model,
                                         all_intents=dataset_labels)

    print(all_examples)
    pickle.dump(all_examples, open(
        fname,
        "wb"))
