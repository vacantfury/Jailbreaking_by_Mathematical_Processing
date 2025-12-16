"""
Evaluation module for assessing LLM responses.

This module provides both class-based and functional APIs for evaluation:

Class-based (recommended for experiments):
    from src.evaluation import Evaluator
    evaluator = Evaluator(model=LLMModel.GPT_4)
    result = evaluator.evaluate_harm(prompt, risk, response)

Functional (for one-off evaluations):
    from src.evaluation import get_harm_score, get_obedience_score
    result = get_harm_score(prompt, risk, response, model=LLMModel.GPT_4)
"""

# All evaluation functionality is in evaluator.py
from .evaluator import (
    # Class-based API (recommended)
    Evaluator,
    
    # Functional API (for one-off evaluations)
    get_harm_score,
    get_obedience_score,
    
    # Data loading functions
    load_prompts_from_file,
    load_and_filter_prompts,
    
    # Helper functions (for advanced use)
    extract_content,
    get_evaluation_template,
    get_obedience_evaluation_template,
    process_evaluation_response,
    process_obedience_evaluation_response,
)

# Constants
from . import constants

__all__ = [
    # Class-based API
    'Evaluator',
    
    # Functional API
    'get_harm_score',
    'get_obedience_score',
    
    # Data loading
    'load_prompts_from_file',
    'load_and_filter_prompts',
    
    # Helper functions
    'extract_content',
    'get_evaluation_template',
    'get_obedience_evaluation_template',
    'process_evaluation_response',
    'process_obedience_evaluation_response',
    
    # Constants module
    'constants',
]

