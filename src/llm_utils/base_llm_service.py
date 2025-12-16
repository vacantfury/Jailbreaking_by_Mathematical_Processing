"""
Base abstract class for LLM services.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class BaseLLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def batch_generate(
        self,
        prompts: List[Tuple[str, str]],
        system_message: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate text responses for multiple prompts.
        
        Args:
            prompts: List of (id, prompt) tuples where id is a unique identifier
            system_message: Optional system message/instruction
            **kwargs: Additional model-specific parameters (temperature, max_tokens, etc.)
        
        Returns:
            List of (id, response) tuples
        """
        raise NotImplementedError
    
    @abstractmethod
    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[str]]]]],
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate responses for multiple chat conversations.
        
        Args:
            conversations: List of (id, messages) tuples, where id is a unique identifier
                and messages is a list of (prompt, image_path) tuples. image_path can be None
                for text-only messages.
            **kwargs: Additional model-specific parameters
        
        Returns:
            List of (id, response) tuples
        """
        raise NotImplementedError
