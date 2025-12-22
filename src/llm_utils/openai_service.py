"""
OpenAI service implementation.
"""
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import base64

from .base_llm_service import BaseLLMService
from .llm_model import LLMModel
from .constants import (
    OPENAI_API_KEY,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    MODELS_USING_MAX_COMPLETION_TOKENS,
    MODELS_WITHOUT_TEMPERATURE_SUPPORT,
)
from ..utils.logger import get_logger
from ..utils import multiprocess_run
from ..utils.constants import PARALLEL_PROCESSING_THRESHOLD

logger = get_logger(__name__)


class OpenAIService(BaseLLMService):
    """Service for OpenAI models (GPT-3.5, GPT-4, etc.)."""
    
    def __init__(self, model: LLMModel, **kwargs):
        """
        Initialize OpenAI service.
        
        Args:
            model: The LLM model to use
            **kwargs: Additional parameters
                - api_key (str): OpenAI API key (optional, will load from env)
                - temperature (float): Sampling temperature (default: from constants.py)
                - max_tokens (int): Maximum tokens to generate (default: from constants.py)
        """
        self.model = model
        self.api_key = kwargs.get('api_key') or OPENAI_API_KEY
        if not self.api_key:
            logger.error("OpenAI API key not found")
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env or pass api_key parameter")
        self.temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)
        self.max_tokens = kwargs.get('max_tokens', DEFAULT_MAX_TOKENS)
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI service with {model.model_id}")
        except ImportError:
            logger.error("OpenAI package not installed")
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def _uses_max_completion_tokens(self) -> bool:
        """
        Check if this model uses max_completion_tokens instead of max_tokens.
        
        Newer models (GPT-4.1+, GPT-5, GPT-O series) use max_completion_tokens.
        Older models (GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o) use max_tokens.
        
        Uses the MODELS_USING_MAX_COMPLETION_TOKENS set from constants for
        accurate model-specific behavior.
        
        Returns:
            True if model uses max_completion_tokens, False if it uses max_tokens
        """
        return self.model in MODELS_USING_MAX_COMPLETION_TOKENS
    
    def _supports_temperature(self) -> bool:
        """
        Check if this model supports custom temperature values.
        
        Some newer models (e.g., GPT-5-Nano) only support the default temperature (1.0)
        and will error if you try to set a custom value.
        
        Uses the MODELS_WITHOUT_TEMPERATURE_SUPPORT set from constants for
        accurate model-specific behavior.
        
        Returns:
            True if model supports custom temperature, False if it only accepts default
        """
        return self.model not in MODELS_WITHOUT_TEMPERATURE_SUPPORT
    
    @staticmethod
    def _prepare_prompt(
        prompt_data: Tuple[str, str],
        system_message: Optional[str]
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Prepare a single prompt for API call (CPU-bound preprocessing).
        
        Args:
            prompt_data: (id, prompt_text) tuple
            system_message: Optional system message
        
        Returns:
            (id, messages) tuple ready for API call
        """
        prompt_id, prompt_text = prompt_data
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt_text})
        return (prompt_id, messages)
    
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
        openai_messages = []
        
        for prompt_text, image_path in messages:
            if image_path is None:
                # Text-only message
                openai_messages.append({
                    "role": "user",
                    "content": prompt_text
                })
            else:
                # Multimodal message with image - encode image (CPU-bound)
                try:
                    with open(image_path, 'rb') as img_file:
                        image_data = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    # Determine image format
                    ext = Path(image_path).suffix.lower()
                    mime_type = {
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.png': 'image/png',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp'
                    }.get(ext, 'image/jpeg')
                    
                    openai_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    })
                except Exception as e:
                    # If image loading fails, send text only
                    logger.warning(f"Image load error for {image_path}: {str(e)}")
                    openai_messages.append({
                        "role": "user",
                        "content": f"{prompt_text} [Image load error: {str(e)}]"
                    })
        
        return (conv_id, openai_messages)
    
    def prepare_batch_prompts(
        self,
        prompts: List[Tuple[str, str]],
        system_message: Optional[str] = None,
    ) -> List[Tuple[str, List[Dict[str, str]]]]:
        """
        Prepare multiple prompts in parallel for batch API submission.
        Automatically uses parallel processing for larger batches.
        
        Args:
            prompts: List of (id, prompt_text) tuples
            system_message: Optional system message
        
        Returns:
            List of (id, formatted_messages) tuples ready for batch API
        """
        # Check if we are already in a daemon process (nested pool is not allowed)
        import multiprocessing
        is_daemon = multiprocessing.current_process().daemon
        
        # Auto-enable parallel processing based on threshold, unless we are already in a daemon
        if len(prompts) < PARALLEL_PROCESSING_THRESHOLD or is_daemon:
            return [self._prepare_prompt(p, system_message) for p in prompts]
        
        # Use parallel processing for CPU-bound preparation
        return multiprocess_run(
            self._prepare_prompt,
            prompts,
            task_type="cpu",
            system_message=system_message
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
        # Check if we are already in a daemon process (nested pool is not allowed)
        import multiprocessing
        is_daemon = multiprocessing.current_process().daemon
        
        # Auto-enable parallel processing based on threshold, unless we are already in a daemon
        if len(conversations) < PARALLEL_PROCESSING_THRESHOLD or is_daemon:
            return [self._prepare_conversation(c) for c in conversations]
        
        # Use parallel processing for CPU-bound preparation (especially image encoding)
        return multiprocess_run(
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
        2. Sequential API calls (or submit to OpenAI's native Batch API)
        
        Note: For large-scale batch processing (100+ prompts), consider submitting
        prepared prompts to OpenAI's native Batch API (50% cost reduction):
        https://platform.openai.com/docs/guides/batch
        
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
        
        # Sequential API calls (or submit prepared batch to OpenAI Batch API)
        results = []
        total = len(prepared)
        for idx, (prompt_id, messages) in enumerate(prepared, 1):
            try:
                logger.debug(f"Processing request {idx}/{total} (ID: {prompt_id})")
                # Use appropriate parameters based on model capabilities
                api_params = {
                    "model": self.model.model_id,
                    "messages": messages,
                }
                
                # Only add temperature if model supports it
                if self._supports_temperature():
                    api_params["temperature"] = temperature
                # else: Use default temperature (1.0) implicitly
                
                # Newer models use max_completion_tokens, older use max_tokens
                if self._uses_max_completion_tokens():
                    api_params["max_completion_tokens"] = max_tokens
                else:
                    api_params["max_tokens"] = max_tokens
                
                response = self.client.chat.completions.create(**api_params)
                
                # Extract response and check for safety filtering
                choice = response.choices[0]
                response_text = choice.message.content
                finish_reason = choice.finish_reason
                
                # Log raw response details for debugging
                logger.debug(f"Prompt {prompt_id}: finish_reason={finish_reason}, content_length={len(response_text) if response_text else 0}")
                
                # Handle empty/None responses due to safety filters
                if not response_text or response_text.strip() == "":
                    filter_msg = f"[LLM response filtered out due to: {finish_reason}"
                    
                    # Check for additional safety information
                    if hasattr(response, 'system_fingerprint'):
                        filter_msg += f", system_fingerprint={response.system_fingerprint}"
                    
                    # Log the full response object for debugging
                    logger.warning(f"Empty response for prompt {prompt_id}. Finish reason: {finish_reason}")
                    logger.debug(f"Full response object: {response.model_dump_json() if hasattr(response, 'model_dump_json') else str(response)}")
                    
                    filter_msg += "]"
                    response_text = filter_msg
                
                results.append((prompt_id, response_text))
                logger.debug(f"Completed request {idx}/{total} (ID: {prompt_id})")
            except Exception as e:
                # Check for fatal model errors (404 Not Found)
                error_str = str(e).lower()
                if "not found" in error_str or "does not exist" in error_str or (hasattr(e, 'status_code') and e.status_code == 404):
                    logger.critical(f"FATAL: Model ID {self.model.model_id} not found/unrecognized.")
                    from src.utils.exceptions import FatalModelError
                    raise FatalModelError(f"Model {self.model.model_id} not found") from e
                    
                error_msg = str(e)
                logger.error(f"OpenAI API error for prompt {prompt_id} (request {idx}/{total}): {error_msg}")
                results.append((prompt_id, f"Error: {error_msg}"))
        
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
        2. Sequential API calls (or submit to OpenAI's native Batch API)
        
        Note: For large-scale batch processing with images, consider submitting
        prepared conversations to OpenAI's native Batch API for efficient processing.
        
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
        
        # Sequential API calls (or submit prepared batch to OpenAI Batch API)
        results = []
        total = len(prepared)
        for idx, (conv_id, openai_messages) in enumerate(prepared, 1):
            try:
                logger.debug(f"Processing conversation {idx}/{total} (ID: {conv_id})")
                # Use appropriate parameters based on model capabilities
                api_params = {
                    "model": self.model.model_id,
                    "messages": openai_messages,
                }
                
                # Only add temperature if model supports it
                if self._supports_temperature():
                    api_params["temperature"] = temperature
                # else: Use default temperature (1.0) implicitly
                
                # Newer models use max_completion_tokens, older use max_tokens
                if self._uses_max_completion_tokens():
                    api_params["max_completion_tokens"] = max_tokens
                else:
                    api_params["max_tokens"] = max_tokens
                
                response = self.client.chat.completions.create(**api_params)
                
                # Extract response and check for safety filtering
                choice = response.choices[0]
                response_text = choice.message.content
                finish_reason = choice.finish_reason
                
                # Log raw response details for debugging
                logger.debug(f"Conversation {conv_id}: finish_reason={finish_reason}, content_length={len(response_text) if response_text else 0}")
                
                # Handle empty/None responses due to safety filters
                if not response_text or response_text.strip() == "":
                    filter_msg = f"[LLM response filtered out due to: {finish_reason}"
                    
                    # Check for additional safety information
                    if hasattr(response, 'system_fingerprint'):
                        filter_msg += f", system_fingerprint={response.system_fingerprint}"
                    
                    # Log the full response object for debugging
                    logger.warning(f"Empty response for conversation {conv_id}. Finish reason: {finish_reason}")
                    logger.debug(f"Full response object: {response.model_dump_json() if hasattr(response, 'model_dump_json') else str(response)}")
                    
                    filter_msg += "]"
                    response_text = filter_msg
                
                results.append((conv_id, response_text))
                logger.debug(f"Completed conversation {idx}/{total} (ID: {conv_id})")
            except Exception as e:
                # Check for fatal model errors (404 Not Found)
                error_str = str(e).lower()
                if "not found" in error_str or "does not exist" in error_str or (hasattr(e, 'status_code') and e.status_code == 404):
                    logger.critical(f"FATAL: Model ID {self.model.model_id} not found/unrecognized.")
                    from src.utils.exceptions import FatalModelError
                    raise FatalModelError(f"Model {self.model.model_id} not found") from e
                    
                error_msg = str(e)
                logger.error(f"OpenAI API error for conversation {conv_id} (request {idx}/{total}): {error_msg}")
                results.append((conv_id, f"Error: {error_msg}"))
        
        return results

