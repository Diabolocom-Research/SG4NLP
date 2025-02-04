import os
import requests
from dotenv import load_dotenv
from langchain_core.outputs import Generation
from langchain_core.runnables import Runnable
from langchain_core.outputs import Generation, GenerationChunk, LLMResult

from typing import (
    Any,
    List,
    Optional,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

def caching_layer_diabolocom(model_name, model_temperature, message, llm, redis_cli,
                  use_redis_caching, project_string=""):
    """
    Implements a simple redis based caching layer.
    :param model_name:
    :param model_temperature: if temp!=0, then it actually hits the end point
    :param message:
    :param llm:
    :param redis_cli:
    :param use_redis_caching:
    :param project_string:
    :return:
    """
    if model_temperature != 0.0 or use_redis_caching is False:
        # output = llm.invoke(message)

        output = llm._generate([message]).generations[0][0].text
        return output

    composite_key = str(model_name) + str(model_temperature) + str(message) + str(project_string)
    cached_response = redis_cli.get(composite_key)
    if cached_response:
        return cached_response
    else:
        output = llm._generate([message]).generations[0][0].text
        redis_cli.set(composite_key, output)
        return output


LLAMA3_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{}<|eot_id|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"""



class DiabolocomLLM(Runnable):

    def __init__(self):
        _ = load_dotenv()
        url = "195.154.74.117" # older url - http://195.154.70.74:30000 (used by mlops) and for r&D http://195.154.74.117:30000
        self.url = os.getenv("SG_SERVER", "http://185.175.111.105:30000") # http://195.154.70.74:30000 ; 195.154.74.117
        self.sglang_token = os.getenv("SGL_ENDPOINT_SECRET", "http://185.175.111.105:30000")
        self.server_url = f"{self.url}/generate"
        self.headers = {"Content-Type": "application/json; charset=utf-8"}
        self.headers["Authorization"] = f"Bearer {self.sglang_token}"

    def get_completion(self,
                       prompt,
                       max_tokens=2048,
                       regex=None,
                       temperature=0.0):
        if type(max_tokens) != int:
            max_tokens = 1280

        data = {'text': prompt,
                "sampling_params": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "regex": regex}}

        response = requests.post(self.server_url, json=data, headers=self.headers).json()
        return response['text']

    def invoke(self, text, max_tokens=None, temperature=0.0, regex=None):
        if not isinstance(text, str):
            text = text.text

        response = self.get_completion(text,
                                       max_tokens=max_tokens,
                                       regex=regex,
                                       temperature=temperature)

        return response

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def validate_environment(cls, values):
        return {}

    @property
    def _identifying_params(self):
        """Get the identifying parameters."""
        return True

    @property
    def _invocation_params(self):
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Diabolocom LLM"

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ):
        choices = [self.invoke(prompts[0])]
        generations = []

        generations.append([
            Generation(text=choices[0],
                       generation_info=dict(finish_reason=None, logprobs=None))
        ])

        llm_output = {"token_usage": 0, "model_name": None}

        return LLMResult(generations=generations, llm_output=llm_output)

    def get_prompt(self,
                   user='',
                   agent='',
                   system="You are helpful assistants, follow instructions"):
        return LLAMA3_TEMPLATE.format(system, user, agent)


if __name__ == '__main__':
    llm = DiabolocomLLM()

    prompt = """<s>[INST]
        Generate 1 sentences that end with 'apple'
        [/INST]
        """

    # prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    # You are an expert in identifying topics and extracting insights from customer communication. Specifically, given a user text, your task is to extract both extractive keywords (direct phrases/terms from the text) and abstractive keywords (higher-level concepts or themes). The text involves emails sent by customers to companies either inquiring about information or submitting complaints. The objective is to utilize these keywords for topic modeling, trend detection, and classification. Note: This prompt will be used for multiple documents, so ensure that abstractive keywords remain cohesive and consistent across different texts.
    #
    # Follow these steps:
    #
    # Extract the extractive keywords directly from the text.
    # Identify the abstractive keywords based on the themes or overarching concepts.
    # Remove extractive keywords that are too specific or generic for generalization.
    # Match each extractive keyword to its corresponding abstractive keyword.
    # Provide a one-line summary that explains the meaning of each abstractive keyword to help match other keywords with similar meanings.
    # Ensure the output is easily parsable in the following format:
    #
    # <\extractive>
    # [Comma-separated extractive keywords]
    # <\extractive>
    #
    # <\abstractive>
    # [Comma-separated abstractive keywords]
    # <\abstractive>
    #
    # <\abstractive-extractive>
    # [(Tuple of extractive keyword - abstractive keyword)]
    # <\abstractive-extractive>
    #
    # <\summary>
    # [One-line summary for each abstractive keyword separated by * delimiter]
    # <\summary>
    #
    #
    # <|eot_id|><|start_header_id|>user<|end_header_id|>
    #
    # Email: I called the customer service number and spoke with Katie before applying for a Value City Card to ask if I apply would I save the 10% and she even asked what item was I was trying to buy, which I told her the Gramercy Park bedroom set, she looked it up and said it was a door buster, and I ask if I apply for the card will I still get the 10% which she said yes. When I placed the order online the 10% was not reflected so I called back to the 1-888-751-8553, and spoke with another young Lady who said that the 10 % did not apply to what I ordered and who did I speak with at first that said that. I did not recall the young ladies name, but asked to speak to a manager to which Olivia took the call. She said forturnaly the call are recored, I said ok good, because I knew what I was quoted. First she said she did not find a call from the numbers I provided, and I told her I was calling from work and we had various lines. I gave her another number to which she found the number I called from 843-398-4000 @ 9:12 was the first call to ask if I would get the 10% off if I applied for the card, to which she said yes, after looking up the item I was trying to get. Ms. Olivia acted as if I was lying about even calling in at first, then when she did see that I called, she said she would listen to the call and reach back out to me. When she reached back out she said it was a misunderstanding and the 10% was not offered. I told her that was not the truth I know that was the reason I even called in and asked to hear the call for myself and she said she could not let me hear the call, but it was my own voice being recorded and if you was in the right why would you not let me hear the call? I am very upset because I would not have applied for this card other than to save the 10%. Now is this is how yall conduct business I do not want no parts of it and I will cancel my order and this card. I know what I was told on the call becasue it was my reason for making the first call and I would not have applied if she said that it did not apply to what I was getting. Now I will take the next step in this matter and go as far as I need to because I know what I was told and what was said on the call. I did not need this card and only applied because she said I would save the additional 10% off. In stead of saying we made a mistake and we will try to fix it, I was made out to be a liar to which I am not happy with. I need somone from cooperate office to reach out to me as soon as possible. I could have use my own credit card and got cash back on my card but only use this one because I was told I could save the additional. Thanks for looking into this matter.
    # "{}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    output = llm._generate([prompt])
    print(output.generations[0][0].text)    