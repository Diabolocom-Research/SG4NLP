import redis
from config import *

from .diabolocom_llm import DiabolocomLLMAdapter
from .llm_abstraction import BaseLLM


class LLMFactory:
    """
    Factory to create an instance of an LLM adapter based on configuration.
    """

    @staticmethod
    def create_llm(llm_config: LLMConfig) -> BaseLLM:
        if llm_config.llm_server == "diabolocom":
            redis_client = redis.Redis(host=llm_config.redis_host,
                                       port=llm_config.redis_port,
                                       decode_responses=True)
            return DiabolocomLLMAdapter(
                model_name=llm_config.model_name,
                redis_client=redis_client,
                project_string=llm_config.project_string,
                temperature=llm_config.temperature,
                use_redis_caching=llm_config.caching
            )
        # elif llm_type == "openai":
        #     return OpenAILLMAdapter(api_key=config.get("api_key"), ...)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_config.llm_type}")
