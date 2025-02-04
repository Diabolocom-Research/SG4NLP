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

    output = llm._generate([prompt])
    print(output.generations[0][0].text)    