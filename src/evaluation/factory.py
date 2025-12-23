from typing import Optional, Dict, Any
from .base_evaluator import BaseEvaluator
from .youjia_evaluation.evaluator import YoujiaEvaluator
from .harmbench_evaluation.evaluator import HarmBenchEvaluator
from src.llm_utils import LLMModel

class EvaluatorFactory:
    """
    Factory for creating evaluators.
    """
    
    @staticmethod
    def create(method: str = "youjia", model: Optional[LLMModel] = None, **kwargs) -> BaseEvaluator:
        """
        Create an evaluator instance.
        
        Args:
            method: Evaluation method ('youjia' or 'harmbench'). Default is 'youjia'.
            model: The LLM model to use for evaluation.
            **kwargs: Additional arguments required by specific evaluators.
            
        Returns:
            An instance of a class inheriting from BaseEvaluator.
            
        Raises:
            ValueError: If the method is unknown.
        """
        method = method.lower().strip()
        
        if method == "youjia":
            return YoujiaEvaluator(model=model, **kwargs)
        elif method == "harmbench":
            return HarmBenchEvaluator(model=model, **kwargs)
        else:
            raise ValueError(f"Unknown evaluation method: {method}. Supported: 'youjia', 'harmbench'")
