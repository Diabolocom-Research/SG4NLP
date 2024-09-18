'''
Endpoint to run the Anyscale and Diabolocom.
Legacy support: Current system uses langchain based endpoints instead of these
custom ones.

After writing the code and publishing the paper, anyscale shut down the service
We recommend using deepinfra or grok for the sae
'''

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
)
import os
import requests
from dotenv import load_dotenv
import sglang as sgl
from dotenv import load_dotenv
from sglang.global_config import global_config
from langchain_community.llms.openai import (
    BaseOpenAI,
    acompletion_with_retry,
    completion_with_retry,
)
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import Generation

from langchain_core.runnables import Runnable


from config import llm_lookup

class AnyscaleLLM(Runnable):

    def __init__(self, llm="mistralai/Mixtral-8x7B-Instruct-v0.1"):
        _ = load_dotenv()
        self.session = requests.Session()

        self.api_base = "https://api.endpoints.anyscale.com/v1"
        self.token = os.getenv('ANYSCALE_TOKEN')
        self.model_name = llm
        self.url = f"{self.api_base}/completions"

    def invoke(self, text, max_tokens=4096, temperature=0.0):
        if not isinstance(text, str):
            text = text.text
        if not isinstance(max_tokens, int):
            max_tokens = 256
        if not isinstance(temperature, int):
            temperature = 0

        body = {
            "model": self.model_name,
            "prompt": text,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        with self.session.post(
                self.url,
                headers={"Authorization": f"Bearer {self.token}"},
                json=body) as resp:
            response = resp.json()['choices'][0]['text']
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
        return "Anyscale LLM"

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


def generate_response(prompt, llm):
    anyscale_llm = [i for i, j in llm_lookup if j == "anyscale"]
    diabolocom_llm = [i for i, j in llm_lookup if j == "diabolocom"]
    if llm in anyscale_llm:
        llm = AnyscaleLLM(llm)
        return llm.invoke(prompt)
    else:
        raise NotImplementedError



if __name__ == '__main__':
    # llm = AnyscaleLLM()

    load_dotenv()
    prompt = """<s>[INST]
    Generate 1 sentences that end with 'apple'
    [/INST]
    """
    # print("*****")
    # print(llm.invoke(prompt))
    # print("*****")
    from langchain_community.llms import Anyscale
    llm = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model = Anyscale(model_name=llm, temperature=0.0)
    model.invoke(prompt)

    # llm = DiabolocomLLM()
    # a = llm._generate([prompt])
    # print("*****")
    # print(a.generations[0][0].text)
