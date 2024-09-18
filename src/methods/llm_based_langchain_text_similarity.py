# currently it will only work for historical literature cross NER.
import random
import re
from tqdm.auto import tqdm
from utils import caching_layer
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.llms import Anyscale
from config import anyscale_llm, gpt_llm, claude_llm
# def caching_layer(model_name, model_temperature, message, llm, redis_cli,
#                   use_redis_caching):
#     if model_temperature != 0.0 or use_redis_caching is False:
#         output = llm.invoke(message)
#         if type(output) != str:
#             output = output.content
#
#         return output
#
#     composite_key = str(model_name) + str(model_temperature) + str(message)
#     cached_response = redis_cli.get(composite_key)
#     if cached_response:
#         return cached_response
#     else:
#         output = llm.invoke(message)
#         if type(output) != str:
#             output = output.content
#         redis_cli.set(composite_key, output)
#         return output


def clean_output(text):
    pattern = r'similarity_rating>(.*?)</similarity_rating'

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)

    # Extract and print the text if the pattern is found
    if match:
        extracted_text = match.group(1).strip()
        digits = re.findall(r'\d+', extracted_text)
        if len(digits) == 0:
            return random.randrange(0,5)
        return int(digits[0])
    else:
        return random.randrange(0,5)

def construct_system_prompt():
    system_prompt = '''Your goal is to rate the similarity between the two texts provided by te user.

         The rating scale is summarized by the following guidelines:

        4, Very Similar -- The two items have very similar meanings and the most important ideas, concepts, or actions in the larger text are represented in the smaller text. Some less important information may be missing, but the smaller text is a very good summary of the larger text.
        3, Somewhat Similar -- The two items share many of the same important ideas, concepts, or actions, but include slightly different details. The smaller text may use similar but not identical concepts (e.g., car vs. vehicle), or may omit a few of the more important ideas present in the larger text.
        2, Somewhat related but not similar -- The two items have dissimilar meaning, but shared concepts, ideas, and actions that are related. The smaller text may use related but not necessary similar concepts (window vs. house) but should still share some overlapping concepts, ideas, or actions with the larger text.
        1, Slightly related -- The two items describe dissimilar concepts, ideas and actions, but may share some small details or domain in common and might be likely to be found together in a longer document on the same topic.
        0, Unrelated -- The two items do not mean the same thing and are not on the same topic.



        After rate the similarity between the text T1 and T2, please provide your output in the following format:

        </similarity_rating>
        [Your similarity score goes here]
        </similarity_rating>


        - Do not provide any additional reasoning.

        Provide similarity rating for

        {string_text}
        '''

    return system_prompt


def generate_predictions(dataset, llm, redis_cli, use_redis_caching, model_name, temperature):
    user_prompt = construct_system_prompt() # after some experiments putting it as user prompt is better for llama. And everything else also works
    preds = []
    for d in tqdm(dataset):
        text = f"T1: {d.text1} \nT2: {d.text2}"

        messages = [
            (
                "system",
                "You are helpful AI assitant",
            ),
            ("human", user_prompt.format(string_text=text)),
        ]
        # time.sleep(3)
        output = caching_layer(model_name=model_name,
                               model_temperature=temperature, message=messages,
                               llm=llm, redis_cli=redis_cli,
                               use_redis_caching=use_redis_caching)
        # print(output)
        pred = clean_output(text=output)

        # pred = process.extract(output, labels, limit=1)[0][0]
        preds.append(pred)

    return preds


def runner(dataset, **kwargs):
    splits = kwargs["splits"]
    llm = kwargs["method"]
    r = kwargs["redis_client"]
    use_redis_caching = kwargs["use_redis_caching"]
    train_pred, dev_pred, test_pred = None, None, None

    if llm in gpt_llm:
        model = ChatOpenAI(
            model=llm,
            temperature=0.0,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    elif llm in anyscale_llm:
        model = Anyscale(model_name=llm, temperature=0.0)
    elif llm in claude_llm:
        model = ChatAnthropic(
            model=llm,
            temperature=0.0,
            max_tokens=1024,
            timeout=None,
            max_retries=2)
    else:
        raise NotImplementedError



    assert splits == "test"
    test_pred = generate_predictions(dataset=dataset["test"], llm=model,
                                redis_cli=r,
                                use_redis_caching=use_redis_caching, model_name=llm,
                                temperature=0.0)


    predictions = {
        "train": train_pred,
        "test": test_pred,
        "dev": dev_pred,
        "extra": None
    }

    return predictions
