from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from src.llm_utils import LLMModel

class BaseEvaluator(ABC):
    """
    Abstract base class for evaluators.
    """
    
    def __init__(self, model: Optional[LLMModel] = None, **kwargs):
        self.model = model
        self.kwargs = kwargs

    @abstractmethod
    def evaluate_jailbreak_effectiveness(
        self, 
        prompts: List[Dict[str, Any]], 
        processed_prompts: List[str], 
        target_model: LLMModel,
        verbose: bool = False,
        baseline_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the effectiveness of a jailbreak attack.

        Args:
            prompts: List of original prompt dictionaries (must contain 'prompt' key).
            processed_prompts: List of processed/jailbroken prompt strings.
            target_model: The target LLM to test.
            verbose: Whether to log detailed progress.
            baseline_results: Optional dictionary containing pre-computed baseline results 
                              (responses and/or evaluations) to avoid re-computation.
                              Structure should match the 'results' output of a previous run.

        Returns:
            Dictionary containing 'results' (list of detailed results per prompt) 
            and 'statistics' (aggregate metrics).
        """
        pass
