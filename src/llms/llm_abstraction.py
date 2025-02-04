from typing import List, Dict
from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Abstract base class defining the LLM interface.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates text based on the provided prompt."""
        pass

    @abstractmethod
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Formats the chat messages according to the LLM's expectations."""
        pass