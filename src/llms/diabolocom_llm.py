import os
import logging
import redis
import requests
from functools import partial
from typing import Any, List, Dict, Optional

from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_core.runnables import Runnable
from langchain_core.outputs import Generation, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from .llm_abstraction import BaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def caching_layer_diabolocom(
        model_name: str,
        model_temperature: float,
        message: str,
        llm: Any,
        redis_cli: redis.Redis,
        use_redis_caching: bool,
        project_string: str = "ner_synthetic_diabolocom"
) -> str:
    """
    Implements a simple Redis-based caching layer.
    If the model temperature is non-zero or caching is disabled,
    the function bypasses the cache and directly queries the LLM.

    :param model_name: Name of the model.
    :param model_temperature: Temperature setting for generation.
    :param message: The prompt/message to send to the LLM.
    :param llm: An instance of the LLM interface.
    :param redis_cli: Redis client for caching.
    :param use_redis_caching: Flag to enable/disable caching.
    :param project_string: Additional string to distinguish cache keys.
    :return: The generated text from the LLM.
    """
    if model_temperature != 0.0 or not use_redis_caching:
        output = llm._generate([message]).generations[0][0].text
        return output

    composite_key = f"{model_name}:{model_temperature}:{message}:{project_string}"
    cached_response = redis_cli.get(composite_key)
    if cached_response:
        # logger.info("Cache hit for key: %s", composite_key)
        return cached_response.decode("utf-8") if isinstance(cached_response, bytes) else cached_response
    else:
        # logger.info("Cache miss for key: %s", composite_key)
        output = llm._generate([message]).generations[0][0].text
        redis_cli.set(composite_key, output)
        return output


LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{}"
)


class DiabolocomLLM(Runnable):
    def __init__(self):
        load_dotenv()
        # Read configuration from environment variables
        self.url = os.getenv("SG_SERVER", "http://185.175.111.105:30000")
        self.sglang_token = os.getenv("SGL_ENDPOINT_SECRET", "")
        if not self.sglang_token:
            logger.warning("SGL_ENDPOINT_SECRET not set in environment variables.")
        self.server_url = f"{self.url}/generate"
        self.headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {self.sglang_token}"
        }
        # Use a session for connection pooling
        self.session = requests.Session()

    def get_completion(
            self,
            prompt: str,
            max_tokens: int = 2048,
            regex: Optional[str] = None,
            temperature: float = 0.0
    ) -> str:
        """
        Sends a prompt to the LLM endpoint and returns the generated text.

        :param prompt: The text prompt to generate from.
        :param max_tokens: Maximum tokens for the generated text.
        :param regex: Optional regex for filtering outputs.
        :param temperature: Sampling temperature.
        :return: Generated text.
        """
        if not isinstance(max_tokens, int):
            logger.warning("max_tokens is not an int, defaulting to 1280")
            max_tokens = 1280

        data = {
            'text': prompt,
            "sampling_params": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "regex": regex
            }
        }
        try:
            response = self.session.post(self.server_url, json=data, headers=self.headers)
            response.raise_for_status()
            response_json = response.json()
            if 'text' not in response_json:
                raise ValueError("Response JSON does not contain 'text' field.")
            return response_json['text']
        except requests.RequestException as e:
            logger.error("Request to LLM server failed: %s", e)
            raise
        except ValueError as ve:
            logger.error("Unexpected response format: %s", ve)
            raise

    def invoke(
            self,
            text: Any,
            max_tokens: Optional[int] = None,
            temperature: float = 0.0,
            regex: Optional[str] = None
    ) -> str:
        """
        Invokes the LLM with the provided text.
        Converts the input to string if necessary.

        :param text: Input prompt, either a string or an object with a 'text' attribute.
        :param max_tokens: Maximum tokens for the response.
        :param temperature: Sampling temperature.
        :param regex: Optional regex for output filtering.
        :return: Generated text.
        """
        if not isinstance(text, str):
            try:
                text = text.text
            except AttributeError:
                logger.error("Input text must be a string or have a 'text' attribute.")
                raise TypeError("Input text must be a string or have a 'text' attribute.")
        return self.get_completion(prompt=text, max_tokens=max_tokens or 2048, regex=regex, temperature=temperature)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    @property
    def _identifying_params(self) -> bool:
        return True

    @property
    def _invocation_params(self) -> bool:
        return True

    @property
    def _llm_type(self) -> str:
        return "Diabolocom LLM"

    def _generate(
            self,
            prompts: List[str],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> LLMResult:
        if not prompts:
            raise ValueError("Prompts list is empty.")
        generated_text = self.invoke(prompts[0])
        generation = Generation(
            text=generated_text,
            generation_info={"finish_reason": None, "logprobs": None}
        )
        llm_output = {"token_usage": 0, "model_name": self._llm_type}
        return LLMResult(generations=[[generation]], llm_output=llm_output)

    def get_prompt(
            self,
            user: str = '',
            agent: str = '',
            system: str = "You are helpful assistants, follow instructions"
    ) -> str:
        """
        Formats the prompt based on a fixed template.

        :param user: The user prompt.
        :param agent: The agent context.
        :param system: The system prompt.
        :return: A formatted prompt string.
        """
        return LLAMA3_TEMPLATE.format(system, user, agent)


class DiabolocomLLMAdapter(BaseLLM):
    """
    LLM adapter for Diabolocom's custom LLM endpoint with caching.
    """

    def __init__(
            self,
            model_name: str,
            redis_client: redis.Redis,
            project_string: str,
            temperature: float = 0.0
    ):
        self.model_name = model_name
        self.redis_client = redis_client
        self.project_string = project_string
        self.temperature = temperature
        self.llm_ep = DiabolocomLLM()
        # Wrap the LLM with caching using functools.partial.
        self.llm = partial(
            caching_layer_diabolocom,
            model_name=self.model_name,
            model_temperature=self.temperature,
            llm=self.llm_ep,
            redis_cli=self.redis_client,
            use_redis_caching=True,
            project_string=self.project_string
        )
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
        except Exception as e:
            logger.error("Failed to load tokenizer: %s", e)
            raise

    def generate(self, prompt: str) -> str:
        """
        Generates text from the LLM using the caching layer.

        :param prompt: The input prompt.
        :return: Generated text.
        """
        return self.llm(message=prompt)

    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Formats messages into a prompt using the tokenizer's chat template.

        :param messages: List of message dictionaries.
        :return: Formatted prompt string.
        """
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            logger.error("Error formatting messages with tokenizer: %s", e)
            raise


if __name__ == '__main__':
    # Example usage of the DiabolocomLLM.
    try:
        llm_instance = DiabolocomLLM()
        prompt = (
            "<s>[INST]\n"
            "Generate 1 sentence that ends with 'apple'\n"
            "[/INST]\n"
        )
        result = llm_instance._generate([prompt])
        print(result.generations[0][0].text)
    except Exception as e:
        logger.error("An error occurred: %s", e)
