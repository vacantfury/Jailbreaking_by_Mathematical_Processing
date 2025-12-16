"""
Google Gemini service implementation.
"""
from typing import List, Tuple, Optional
import google.genai as genai
from .base_llm_service import BaseLLMService
from .llm_model import LLMModel
from .constants import (
    GOOGLE_API_KEY,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class GoogleService(BaseLLMService):
    """Service for Google Gemini models."""

    def __init__(self, model: LLMModel, **kwargs):
        """
        Initialize Google Gemini service.

        Args:
            model: The LLM model to use
            **kwargs: Additional parameters
                - api_key (str): Google API key (optional, will load from env)
                - temperature (float): Sampling temperature (default: from constants.py)
                - max_tokens (int): Maximum tokens to generate (default: from constants.py)
        """
        self.model = model
        self.api_key = kwargs.get('api_key') or GOOGLE_API_KEY
        if not self.api_key:
            logger.error("Google API key not found")
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY in .env or pass api_key parameter")
        self.temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)
        self.max_tokens = kwargs.get('max_tokens', DEFAULT_MAX_TOKENS)

        # Initialize Google Generative AI client (new SDK)
        try:
            # New SDK uses Client() directly, not configure()
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized Google service with {model.model_id}")
        except ImportError:
            logger.error("Google Generative AI package not installed")
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
    
    def batch_generate(
        self,
        prompts: List[Tuple[str, str]],
        system_message: Optional[str] = None,
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate text responses for multiple prompts.

        Args:
            prompts: List of (id, prompt) tuples
            system_message: Optional system message (will be prepended to each prompt)
            **kwargs: Additional parameters
                - temperature: Sampling temperature (default: from instance init or constants.py)
                - max_tokens: Maximum tokens to generate (default: from instance init or constants.py)
                - top_p: Nucleus sampling parameter (default: from constants.py, only used when temperature > 0)

        Returns:
            List of (id, response) tuples. On errors, returns (id, error_message_string).
        """
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', DEFAULT_TOP_P)

        results = []
        total = len(prompts)

        import time
        import random
        
        for idx, (prompt_id, prompt_text) in enumerate(prompts, 1):
            max_retries = 5
            for attempt in range(max_retries + 1):
                try:
                    # Prepend system message if provided
                    full_prompt = prompt_text
                    if system_message:
                        full_prompt = f"{system_message}\n\n{prompt_text}"

                    logger.debug(f"Processing request {idx}/{total} (ID: {prompt_id})")

                    # Configure generation parameters (new SDK)
                    gen_config_dict = {
                        'temperature': temperature,
                        'max_output_tokens': max_tokens,
                    }

                    # Only set top_p if temperature > 0 (sampling enabled)
                    if temperature > 0:
                        gen_config_dict['top_p'] = top_p

                    # Generate response using new SDK: client.models.generate_content()
                    response = self.client.models.generate_content(
                        model=self.model.model_id,
                        contents=full_prompt,
                        config=gen_config_dict
                    )
                    
                    # Extract response text
                    response_text = response.text
                    finish_reason = response.candidates[0].finish_reason if response.candidates else None
                    
                    # Log raw response details for debugging
                    logger.debug(f"Prompt {prompt_id}: finish_reason={finish_reason}, content_length={len(response_text) if response_text else 0}")
                    
                    # Handle empty/None responses due to safety filters
                    if not response_text or response_text.strip() == "":
                        filter_msg = f"[LLM response filtered out due to: {finish_reason}"
                        
                        # Check for safety ratings
                        if response.candidates and response.candidates[0].safety_ratings:
                            safety_info = []
                            for rating in response.candidates[0].safety_ratings:
                                if rating.probability.name != "NEGLIGIBLE":
                                    safety_info.append(f"{rating.category.name}={rating.probability.name}")
                            if safety_info:
                                filter_msg += f", safety_ratings={', '.join(safety_info)}"
                        
                        # Log the full response object for debugging
                        logger.warning(f"Empty response for prompt {prompt_id}. Finish reason: {finish_reason}")
                        logger.debug(f"Full response object: {response}")
                        
                        filter_msg += "]"
                        response_text = filter_msg
                    
                    results.append((prompt_id, response_text))
                    logger.debug(f"Completed request {idx}/{total} (ID: {prompt_id})")
                    break # Success, exit retry loop
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "503" in error_msg or "UNAVAILABLE" in error_msg:
                        if attempt < max_retries:
                            wait_time = (2 ** attempt) + random.uniform(0, 1)
                            logger.warning(f"Rate limit hit for prompt {prompt_id}. Retrying in {wait_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                            time.sleep(wait_time)
                            continue
                    
                    # If not 429 or max retries reached
                    logger.error(f"Google API error for prompt {prompt_id} (request {idx}/{total}): {error_msg}")
                    results.append((prompt_id, f"Error: {error_msg}"))
                    break
        
        return results
    
    def batch_chat(
        self,
        conversations: List[Tuple[str, List[Tuple[str, Optional[str]]]]],
        **kwargs
    ) -> List[Tuple[str, str]]:
        """
        Generate responses for multiple chat conversations.

        Args:
            conversations: List of (id, messages) tuples, where messages is
                a list of (prompt, image_path) tuples
            **kwargs: Additional parameters
                - temperature: Sampling temperature (default: from instance init or constants.py)
                - max_tokens: Maximum tokens to generate (default: from instance init or constants.py)
                - top_p: Nucleus sampling parameter (default: from constants.py, only used when temperature > 0)

        Returns:
            List of (id, response) tuples. On errors, returns (id, error_message_string).
        """
        temperature = kwargs.get('temperature', self.temperature)
        max_tokens = kwargs.get('max_tokens', self.max_tokens)
        top_p = kwargs.get('top_p', DEFAULT_TOP_P)

        results = []
        total = len(conversations)

        for idx, (conv_id, messages) in enumerate(conversations, 1):
            try:
                logger.debug(f"Processing conversation {idx}/{total} (ID: {conv_id})")

                # Build conversation history
                conversation_parts = []
                for prompt_text, image_path in messages:
                    if image_path is None:
                        # Text-only message
                        conversation_parts.append(prompt_text)
                    else:
                        # Multimodal message with image
                        try:
                            import PIL.Image
                            img = PIL.Image.open(image_path)
                            conversation_parts.append(img)
                            conversation_parts.append(prompt_text)
                        except Exception as e:
                            # If image loading fails, send text only
                            logger.warning(f"Image load error for {image_path}: {str(e)}")
                            conversation_parts.append(f"{prompt_text} [Image load error: {str(e)}]")

                # Configure generation parameters (new SDK)
                gen_config_dict = {
                    'temperature': temperature,
                    'max_output_tokens': max_tokens,
                }

                # Only set top_p if temperature > 0 (sampling enabled)
                if temperature > 0:
                    gen_config_dict['top_p'] = top_p

                # Generate response using new SDK: client.models.generate_content()
                response = self.client.models.generate_content(
                    model=self.model.model_id,
                    contents=conversation_parts,
                    config=gen_config_dict
                )
                
                # Extract response text
                response_text = response.text
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                
                # Log raw response details for debugging
                logger.debug(f"Conversation {conv_id}: finish_reason={finish_reason}, content_length={len(response_text) if response_text else 0}")
                
                # Handle empty/None responses due to safety filters
                if not response_text or response_text.strip() == "":
                    filter_msg = f"[LLM response filtered out due to: {finish_reason}"
                    
                    # Check for safety ratings
                    if response.candidates and response.candidates[0].safety_ratings:
                        safety_info = []
                        for rating in response.candidates[0].safety_ratings:
                            if rating.probability.name != "NEGLIGIBLE":
                                safety_info.append(f"{rating.category.name}={rating.probability.name}")
                        if safety_info:
                            filter_msg += f", safety_ratings={', '.join(safety_info)}"
                    
                    # Log the full response object for debugging
                    logger.warning(f"Empty response for conversation {conv_id}. Finish reason: {finish_reason}")
                    logger.debug(f"Full response object: {response}")
                    
                    filter_msg += "]"
                    response_text = filter_msg
                
                results.append((conv_id, response_text))
                logger.debug(f"Completed conversation {idx}/{total} (ID: {conv_id})")
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Google API error for conversation {conv_id} (request {idx}/{total}): {error_msg}")
                results.append((conv_id, f"Error: {error_msg}"))
        
        return results

