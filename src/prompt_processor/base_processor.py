"""
Base processor class and helper functions for all prompt processors.
"""
from abc import ABC, abstractmethod
from typing import Optional, List

from src.llm_utils import LLMModel
from src.utils.logger import get_logger
from src.utils import parallel_map
from src.utils.constants import PARALLEL_PROCESSING_THRESHOLD

logger = get_logger(__name__)


class BaseProcessor(ABC):
    """
    Abstract base class for all prompt processors.
    
    Processors transform prompts using different techniques (LLM-based or rule-based).
    Each processor must implement process() for single prompt transformation.
    The default batch_process() uses multiprocessing for efficiency.
    
    All processors accept an optional 'model' parameter for consistency,
    even if they don't use it (e.g., rule-based processors).
    
    If no model is provided for LLM-based processors, they will use 
    DEFAULT_PROCESSING_MODEL from processors/constants.py (GPT-4o).
    """
    
    def __init__(self, model: Optional[LLMModel] = None, **kwargs):
        """
        Initialize the processor.
        
        Args:
            model: Optional LLM model (used by LLM-based processors).
                  If None, LLM-based processors will use DEFAULT_PROCESSING_MODEL.
            **kwargs: Processor-specific parameters
        """
        self.model = model
    
    @abstractmethod
    def process(self, prompt: str, **kwargs) -> str:
        """
        Process a single prompt.
        
        Args:
            prompt: The prompt to process
            **kwargs: Processor-specific parameters
            
        Returns:
            Processed prompt or response
        """
        pass
    
    def batch_process(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Process multiple prompts efficiently.
        
        Default implementation uses multiprocessing for CPU-bound processors.
        LLM-based processors should override this to use API batch calls.
        
        Args:
            prompts: List of prompts to process
            **kwargs: Processor-specific parameters
            
        Returns:
            List of processed prompts/responses
        """
        if not prompts:
            logger.warning("No prompts to process")
            return []
        
        logger.info(f"Batch processing {len(prompts)} prompts")
        
        # For large batches, use multiprocessing
        if len(prompts) >= PARALLEL_PROCESSING_THRESHOLD:
            logger.info("Using parallel processing (multiprocessing)")
            
            try:
                # Force sequential execution to avoid nested multiprocessing
                # (since tasks are now executed in parallel)
                results = parallel_map(
                    self._process_with_error_handling,
                    prompts,
                    task_type="cpu",
                    sequential=True,  # Forced sequential execution
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Parallel processing failed: {e}, falling back to sequential")
                results = self._sequential_process(prompts, **kwargs)
        else:
            # Sequential processing for small batches
            logger.info("Using sequential processing (small batch)")
            results = self._sequential_process(prompts, **kwargs)
        
        logger.info(f"Batch processing complete: {len(results)} results")
        return results
    
    def _sequential_process(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Process prompts sequentially (fallback or small batches).
        
        Args:
            prompts: List of prompts to process
            **kwargs: Processor-specific parameters
            
        Returns:
            List of processed prompts
        """
        results = []
        for prompt in prompts:
            try:
                result = self.process(prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing prompt: {str(e)}")
                results.append(f"Error: {str(e)}")
        return results
    
    def _process_with_error_handling(self, prompt: str, **kwargs) -> str:
        """
        Wrapper for parallel processing with error handling.
        
        Args:
            prompt: The prompt to process
            **kwargs: Processor-specific parameters
            
        Returns:
            Processed result or error message
        """
        try:
            return self.process(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Error in parallel processing: {str(e)}")
            return f"Error: {str(e)}"


def split_into_parts(words: list[str], num_parts: int) -> list[list[str]]:
    """
    Split a list of words into parts with smart boundary detection.
    
    This function tries to split at natural boundaries (sentences, phrases)
    rather than cutting words arbitrarily.
    
    Args:
        words: List of words to split
        num_parts: Number of parts to create
        
    Returns:
        List of parts, where each part is a list of words
    """
    # Import constants locally to avoid circular imports
    from .processors.constants import (
        BOUNDARY_WINDOW_PERCENT,
        SENTENCE_BOUNDARY_SCORE,
        PHRASE_BOUNDARY_SCORE,
        WORD_BOUNDARY_SCORE,
        MIN_CONTENT_WORD_LENGTH,
        SENTENCE_ENDERS,
        PHRASE_ENDERS
    )
    
    if len(words) <= num_parts:
        return [[word] for word in words[:num_parts]]
    
    # Calculate target size for each part
    target_size = len(words) / num_parts
    
    parts = []
    current_idx = 0
    
    for i in range(num_parts - 1):
        target_end = int((i + 1) * target_size)
        end_idx = target_end
        
        # Look for a good boundary within a reasonable range
        window_size = max(1, int(target_size * BOUNDARY_WINDOW_PERCENT))
        best_boundary = end_idx
        best_score = -1
        
        # Search within the window for an optimal boundary
        search_start = max(current_idx + 1, end_idx - window_size)
        search_end = min(len(words) - 1, end_idx + window_size)
        
        for j in range(search_start, search_end + 1):
            if j <= current_idx or j >= len(words):
                continue
            
            current_word = words[j - 1]
            score = 0
            
            # Prefer sentence boundaries over phrase boundaries
            if any(current_word.endswith(end) for end in SENTENCE_ENDERS):
                score = SENTENCE_BOUNDARY_SCORE
            elif any(current_word.endswith(end) for end in PHRASE_ENDERS):
                score = PHRASE_BOUNDARY_SCORE
            elif len(current_word) > MIN_CONTENT_WORD_LENGTH:
                score = WORD_BOUNDARY_SCORE
            
            if score > best_score:
                best_score = score
                best_boundary = j
        
        # Use the best boundary found
        if best_score > 0:
            end_idx = best_boundary
        
        parts.append(words[current_idx:end_idx])
        current_idx = end_idx
    
    # Add the last part (remaining words)
    parts.append(words[current_idx:])
    
    return parts

