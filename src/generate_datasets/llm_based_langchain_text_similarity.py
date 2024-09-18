import os
import re
import pickle
import random
import traceback
from pathlib import Path
from tqdm.auto import tqdm
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Anyscale
from src.config import TextSimilarityDataPoint, anyscale_llm, gpt_llm, claude_llm


# def clean_output(text):
#     pattern = r'<generated_text>(.*?)</generated_text>'
#
#     # Search for the pattern in the text
#     match = re.search(pattern, text, re.DOTALL)
#
#     # Extract and print the text if the pattern is found
#     if match:
#         extracted_text = match.group(1).strip()
#         return extracted_text
#     else:
#         return "No text found between <generated_text> tags."


def clean_output(text):
    pattern = r'generated_text>(.*?)</generated_text'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # Extract and print the text if the pattern is found
    if match:
        extracted_text = match.group(1).strip()
        return [e.replace("T1:", "").replace("T2:", "").strip() for e in extracted_text.split("\n")]
    else:
        return -1

def find_representative_examples(candidate_labels, dataset, k):
    def find_example_with_score(dataset, score):
        all_examples = []
        for i in dataset:
            if round(i.score) == score:
                all_examples.append(i)

        return all_examples

    examples = []

    for score in candidate_labels:
        examples += find_example_with_score(dataset, score)[:k]

    final_string = ""
    random.shuffle(examples)
    for e in examples:
        string = f"Score: {round(e.score)}\n"
        string = string + "T1: " + e.text1 + "\n"
        string = string + "T2: " + e.text2 + "\n"
        final_string = final_string + string + "\n"

    return final_string


def generate_examples(example_string, number_of_examples, llm, dataset_name):
    system_prompt_headlines = f'''You are an AI assistant tasked with generating to short headlines for evaluating text similairty systems.

    The rating scale is summarized by the following guidelines:

        4, Very Similar -- The two items have very similar meanings and the most important ideas, concepts, or actions in the larger text are represented in the smaller text. Some less important information may be missing, but the smaller text is a very good summary of the larger text.
        3, Somewhat Similar -- The two items share many of the same important ideas, concepts, or actions, but include slightly different details. The smaller text may use similar but not identical concepts (e.g., car vs. vehicle), or may omit a few of the more important ideas present in the larger text.
        2, Somewhat related but not similar -- The two items have dissimilar meaning, but shared concepts, ideas, and actions that are related. The smaller text may use related but not necessary similar concepts (window vs. house) but should still share some overlapping concepts, ideas, or actions with the larger text.
        1, Slightly related -- The two items describe dissimilar concepts, ideas and actions, but may share some small details or domain in common and might be likely to be found together in a longer document on the same topic.
        0, Unrelated -- The two items do not mean the same thing and are not on the same topic.

    Your goal is to create two headlines that is representative of the score provided by the user. Some examples are

    {example_string}


    Follow these instructions to generate the text:

    1. Generate a text with the score provided by the user.
    2. The size of text should be similar to the examples provided above following the headline theme.
    3. Be creative and keep the text natural, following the style and tone of the examples provided above.
    4. Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).
    5. Make absolutely sure the generated text has the same scoore as provided by the user



    After generating the text, please provide your output in the following format:

    </generated_text>
    T1: The first text goes here
    T2: The second text goes here
    </generated_text>

    Do not generate anything apart from the text, and strictly follow the structure defined above.

    Remember, the goal is to create realistic-looking sample that could be used to test sentence similarity systems. Make sure your text sounds natural and incorporates the theme of the examples above.
    '''


    system_prompt_twitter = f'''You are an AI assistant tasked with generating to short tweet about news for evaluating text similairty systems.

The rating scale is summarized by the following guidelines:

    4, Very Similar -- The two items have very similar meanings and the most important ideas, concepts, or actions in the larger text are represented in the smaller text. Some less important information may be missing, but the smaller text is a very good summary of the larger text.
    3, Somewhat Similar -- The two items share many of the same important ideas, concepts, or actions, but include slightly different details. The smaller text may use similar but not identical concepts (e.g., car vs. vehicle), or may omit a few of the more important ideas present in the larger text.
    2, Somewhat related but not similar -- The two items have dissimilar meaning, but shared concepts, ideas, and actions that are related. The smaller text may use related but not necessary similar concepts (window vs. house) but should still share some overlapping concepts, ideas, or actions with the larger text.
    1, Slightly related -- The two items describe dissimilar concepts, ideas and actions, but may share some small details or domain in common and might be likely to be found together in a longer document on the same topic.
    0, Unrelated -- The two items do not mean the same thing and are not on the same topic.
    
Your goal is to create two tweets that is representative of the score provided by the user. Some examples are
 
 {example_string}


Follow these instructions to generate the tweet:

1. Generate the tweets with the score provided by the user.
2. The size of text shoule be similar to the examples provided above following the twitter news theme.
3. Be creative and keep the text natural, following the style and tone of the examples provided above.
4. Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).
5. Make absolutely sure the generated text has the same scoore as provided by the user



After generating the text, please provide your output in the following format:

</generated_text>
T1: The first text goes here
T2: The second text goes here
</generated_text>

Do not generate anything apart from the text, and strictly follow the structure defined above.

Remember, the goal is to create realistic-looking sample that could be used to test sentence similarity systems. Make sure your text sounds natural and incorporates the theme of the examples above.

'''

    if dataset_name == "headlines":
        system_prompt = system_prompt_headlines
    elif dataset_name == "tweet-news":
        system_prompt = system_prompt_twitter
    else:
        raise NotImplementedError

    all_examples = []
    counter = 0
    pbar = tqdm(total=number_of_examples)
    while len(all_examples) < number_of_examples:
        if counter > number_of_examples + 200:
            break
        counter += 1

        try:
            score = random.randrange(0, 5)
            user_prompt = f'''Score: {score}'''
            messages = [
                            (
                                "system",
                                system_prompt.format(example_string=example_string),
                            ),
                            ("human", user_prompt),
                        ]

            output = llm.invoke(messages)
            if type(output) != str:
                output = output.content

            output = clean_output(output)
            if type(output) == list and len(output) == 2:
                all_examples.append(TextSimilarityDataPoint(text1=output[0], text2=output[1], score=score))
            pbar.update(1)
        except:
            traceback.print_exc()
            continue

    return all_examples


def generate_dataset(dataset, llm, **kwargs):
    dataset_theme = dataset["extra"]["theme"]
    dataset_name = dataset["extra"]["name"]
    k = kwargs["number_of_examples_per_score"]
    datasaving_string = kwargs["dataset_string"]
    dataset_score = dataset["extra"]["labels"]



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
    example_string = find_representative_examples(dataset_score, dataset["test"], k)
    fname =  f"../data/generated/{datasaving_string}.pkl"

    # Step 3: Generate examples
    if kwargs["caching"]:
        # fname = f"../generation/{dataset_name}_{llm.replace('/', '_')}_{kwargs['number_of_examples_per_intent']}_{kwargs['number_of_examples']}.pkl"
        if os.path.isfile(fname):
            all_examples = pickle.load(open(fname, "rb"))
        else:
            all_examples = generate_examples(example_string=example_string,
                                             number_of_examples=kwargs[
                                                 'number_of_examples'], llm=model,
                                             dataset_name=dataset_name)
            pickle.dump(all_examples, open(fname, "wb"))
    else:
        all_examples = generate_examples(example_string=example_string,
                                         number_of_examples=kwargs[
                                             'number_of_examples'], llm=model,
                                         dataset_name=dataset_name)

    print(len(all_examples))
    pickle.dump(all_examples, open(
        fname,
        "wb"))
