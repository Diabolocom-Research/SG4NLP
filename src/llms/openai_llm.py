import logging
import os
from functools import partial
from typing import Any, List, Dict, Optional

from openai import OpenAI
import redis
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import Generation, LLMResult
from langchain_core.runnables import Runnable
# For this example, we use a simple join formatter instead of a dedicated tokenizer.
# You might replace this with your preferred message formatting.
from llms.llm_abstraction import BaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Caching Layer for OpenAI GPT
# -----------------------------------------------------------------------------

def caching_layer_openai(
        model_name: str,
        model_temperature: float,
        message: List,
        llm: Any,
        redis_cli: redis.Redis,
        use_redis_caching: bool,
        project_string: str = "ner_synthetic_openai"
) -> str:
    """
    Implements a simple Redis-based caching layer for OpenAI GPT.
    If the temperature is non-zero or caching is disabled,
    the cache is bypassed.

    Note that this caching layer is different than the other. Here message is a list!!
    """
    if model_temperature != 0.0 or not use_redis_caching:
        output = llm.generate(message)
        return output

    composite_key = f"{model_name}:{model_temperature}:{str(message)}:{project_string}"
    cached_response = redis_cli.get(composite_key)
    if cached_response:
        return cached_response.decode("utf-8") if isinstance(cached_response, bytes) else cached_response
    else:
        output = llm.generate(message)
        redis_cli.set(composite_key, output)
        return output


class OpenAIGPTLLM:
    def __init__(self, model_name, temperature):
        load_dotenv()
        # Read configuration from environment variables
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set in environment variables.")
        self.client = OpenAI(
            api_key=self.api_key,  # This is the default and can be omitted
        )
        self.temperature = temperature


        # Additional parameters (like max tokens) can be set per invocation if needed.

    def generate(self, messages):
        """

        [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]


        """
        assert type(messages) == list
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )

        return completion.choices[0].message.content

# -----------------------------------------------------------------------------
# OpenAIGPTLLMAdapter: Adapter for Integration with Your Code Base
# -----------------------------------------------------------------------------

class OpenAIGPTLLMAdapter(BaseLLM):
    """
    LLM adapter for OpenAI's GPT models with optional Redis caching.
    """

    def __init__(
            self,
            model_name: str,
            redis_client: redis.Redis,
            project_string: str,
            temperature: float = 0.0,
            use_redis_caching: bool = True
    ):
        self.model_name = model_name
        self.redis_client = redis_client
        self.project_string = project_string
        self.temperature = temperature
        self.llm_ep = OpenAIGPTLLM(model_name=self.model_name, temperature=self.temperature)
        # Wrap the LLM with caching using functools.partial.
        self.llm = partial(
            caching_layer_openai,
            model_name=self.model_name,
            model_temperature=self.temperature,
            llm=self.llm_ep,
            redis_cli=self.redis_client,
            use_redis_caching=use_redis_caching,
            project_string=self.project_string
        )
        # For formatting messages, we use a simple join of message components.

    def generate(self, prompt: List[Dict[str, str]]) -> str:
        """
        Generates text using the caching layer or via API!
        """
        return self.llm(message=prompt)

    def format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Formats a list of message dictionaries into a single string prompt.

              [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]



            messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_user_prompt},
    ]


        """
        return messages

# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    try:
        # Test the OpenAIGPTLLM directly.
        openai_llm = OpenAIGPTLLM(model_name="gpt-4o-mini", temperature=0.0)
        # prompt = (
        #     "System: You are a helpful assistant.\n"
        #     "User: Generate one sentence that ends with 'apple'.\n"
        #     "Assistant: "
        # )

        prompt = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Generate one sentence that ends with 'apple'."},
        ]
        result = openai_llm.generate(prompt)
        print("Direct LLM output:\n", result)
    except Exception as e:
        logger.error("An error occurred: %s", e)
