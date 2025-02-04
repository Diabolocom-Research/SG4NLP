import redis
from typing import List, Any, Dict
from .llm_abstraction import BaseLLM
from .diabolocom_llm import DiabolocomLLMAdapter
class LLMFactory:
    """
    Factory to create an instance of an LLM adapter based on configuration.
    """
    @staticmethod
    def create_llm(llm_type: str, config: Dict[str, Any]) -> BaseLLM:
        if llm_type == "diabolocom":
            redis_client = redis.Redis(host=config.get("redis_host", "localhost"),
                                       port=config.get("redis_port", 6379),
                                       decode_responses=True)
            return DiabolocomLLMAdapter(
                model_name=config.get("model_name", "llama3"),
                redis_client=redis_client,
                project_string=config.get("project_string", "test"),
                temperature=config.get("temperature", 0.0)
            )
        # elif llm_type == "openai":
        #     return OpenAILLMAdapter(api_key=config.get("api_key"), ...)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
