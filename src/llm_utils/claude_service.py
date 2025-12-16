"""
Anthropic Claude service implementation.
"""
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import base64

from .base_llm_service import BaseLLMService
from .llm_model import LLMModel
from .constants import (
    ANTHROPIC_API_KEY,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
from ..utils.logger import get_logger
from ..utils import parallel_map
from ..utils.constants import PARALLEL_PROCESSING_THRESHOLD

logger = get_logger(__name__)


class ClaudeService(BaseLLMService):
    """Service for Anthropic Claude models."""
    
    def __init__(self, model: LLMModel, **kwargs):
        """
        Initialize Claude service.
        
        Args:
            model: The LLM model to use
            **kwargs: Additional parameters
                - api_key (str): Anthropic API key (optional, will load from env)
                - temperature (float): Sampling temperature (default: from constants.py)
                - max_tokens (int): Maximum tokens to generate (default: from constants.py)
        """
        self.model = model
        self.api_key = kwargs.get('api_key') or ANTHROPIC_API_KEY
        if not self.api_key:
            logger.error("Anthropic API key not found")
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY in .env or pass api_key parameter")
        self.temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)
        self.max_tokens = kwargs.get('max_tokens', DEFAULT_MAX_TOKENS)
        
        # Initialize Anthropic client
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            logger.info(f"Initialized Claude service with {model.model_id}")
        except ImportError:
            logger.error("Anthropic package not installed")
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    @staticmethod
    def _prepare_prompt(
        prompt_data: Tuple[str, str],
        system_message: Optional[str]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare a single prompt for API call (CPU-bound preprocessing).
        
        Args:
            prompt_data: (id, prompt_text) tuple
            system_message: Optional system message
        
        Returns:
            (id, params_dict) tuple ready for API call
        """
        prompt_id, prompt_text = prompt_data
        params = {
            "messages": [{"role": "user", "content": prompt_text}]
        }
        if system_message:
            params["system"] = system_message
        return (prompt_id, params)
    
    @staticmethod
    def _prepare_conversation(
        conversation_data: Tuple[str, List[Tuple[str, Optional[str]]]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Prepare a single conversation for API call (CPU-bound preprocessing).
        Handles image encoding in parallel.
        
        Args:
            conversation_data: (id, messages) tuple with (text, image_path) messages
        
        Returns:
            (id, formatted_messages) tuple ready for API call
        """
        conv_id, messages = conversation_data
        anthropic_messages = []
        
        for prompt_text, image_path in messages:
            if image_path is None:
                # Text-only message
                anthropic_messages.append({
                    "role": "user",
                    "content": prompt_text
                })
            else:
                # Multimodal message with image - encode image (CPU-bound)
                try:
                    with open(image_path, 'rb') as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Determine media type
                    ext = Path(image_path).suffix.lower()
                    media_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/jpeg')
                    
                    anthropic_messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data
                                }
                            },
                            {"type": "text", "text": prompt_text}
                        ]
                    })
                except Exception as e:
                    # If image loading fails, send text only
                    logger.warning(f"Image load error for {image_path}: {str(e)}")
                    anthropic_messages.append({
                        "role": "user",
                        "content": f"{prompt_text} [Image load error: {str(e)}]"
                    })
        
        return (conv_id, anthropic_messages)
    
    def prepare_batch_prompts(
        self,
        prompts: List[Tuple[str, str]],
        system_message: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Prepare multiple prompts in parallel for batch API submission.
        Automatically uses parallel processing for larger batches.
        
        Args:
            prompts: List of (id, prompt_text) tuples
            system_message: Optional system message
        
        Returns:
            List of (id, params_dict) tuples ready for batch API
        """
        # Auto-enable parallel processing based on threshold, unless we are already in a daemon
        import multiprocessing
        is_daemon = multiprocessing.current_process().daemon
        
        if len(prompts) < PARALLEL_PROCESSING_THRESHOLD or is_daemon:
            return [self._prepare_prompt(p, system_message) for p in prompts]
        
        # Use parallel processing for CPU-bound preparation
        from functools import partial
        prepare_func = partial(self._prepare_prompt, system_message=system_message)
        return parallel_map(
            prepare_func,
            prompts,
            task_type="cpu"
        )
    
    def prepare_batch_conversations(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[str]]]]],
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Prepare multiple conversations in parallel for batch API submission.
        This is especially useful when processing images (CPU-bound encoding).
        Automatically uses parallel processing for larger batches.
        
        Args:
            conversations: List of (id, messages) tuples
        
        Returns:
            List of (id, formatted_messages) tuples ready for batch API
        """
        # Auto-enable parallel processing based on threshold, unless we are already in a daemon
        import multiprocessing
        is_daemon = multiprocessing.current_process().daemon
        
        if len(conversations) < PARALLEL_PROCESSING_THRESHOLD or is_daemon:
            return [self._prepare_conversation(c) for c in conversations]
        
        # Use parallel processing for CPU-bound preparation (especially image encoding)
        return parallel_map(
            self._prepare_conversation,
            conversations,
            task_type="cpu"
        )
    
    def batch_generate(
        self,
        prompts: List[Tuple[str, str]],
        system_message: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate text responses for multiple prompts with automatic parallel preprocessing.
        
        Automatic optimizations:
        1. Parallel preprocessing for message formatting (10+ prompts)
        2. Sequential API calls (or submit to Anthropic's Message Batches API)
        
        Note: For large-scale batch processing (100+ prompts), consider submitting
        prepared prompts to Anthropic's Message Batches API (50% cost reduction):
        https://docs.anthropic.com/en/docs/build-with-claude/message-batches
        
        Args:
            prompts: List of (id, prompt) tuples
            system_message: Optional system message
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            List of (id, response) tuples
        """
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        # Automatic parallel preprocessing (uses threshold from constants)
        prepared = self.prepare_batch_prompts(prompts, system_message)
        
        # Sequential API calls (or submit prepared batch to Anthropic Message Batches API)
        results = []
        for prompt_id, params_dict in prepared:
            try:
                params = {
                    "model": self.model.model_id,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    **params_dict
                }
                
                response = self.client.messages.create(**params)
                response_text = response.content[0].text
                results.append((prompt_id, response_text))
            except Exception as e:
                logger.error(f"Claude API error for prompt {prompt_id}: {str(e)}")
                results.append((prompt_id, f"Error: {str(e)}"))
        
        return results
    
    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[str]]]]],
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate responses for multiple chat conversations with automatic parallel preprocessing.
        
        Automatic optimizations:
        1. Parallel preprocessing for image encoding (5+ conversations)
        2. Sequential API calls (or submit to Anthropic's Message Batches API)
        
        Note: For large-scale batch processing with images, consider submitting
        prepared conversations to Anthropic's Message Batches API for efficient processing.
        
        Args:
            conversations: List of (id, messages) tuples, where messages is
                a list of (prompt, image_path) tuples
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            List of (id, response) tuples
        """
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        
        # Automatic parallel preprocessing (uses threshold from constants)
        prepared = self.prepare_batch_conversations(conversations)
        
        # Sequential API calls (or submit prepared batch to Anthropic Message Batches API)
        results = []
        for conv_id, anthropic_messages in prepared:
            try:
                response = self.client.messages.create(
                    model=self.model.model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=anthropic_messages
                )
                response_text = response.content[0].text
                results.append((conv_id, response_text))
            except Exception as e:
                logger.error(f"Claude API error for conversation {conv_id}: {str(e)}")
                results.append((conv_id, f"Error: {str(e)}"))
        
        return results

