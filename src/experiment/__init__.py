"""
Experiment package for LLM safety and jailbreak testing.

This package provides tools for:
- Running systematic experiments with harmful prompts
- Testing multiple LLM models
- Evaluating responses for safety and obedience
- Saving structured results for analysis
"""

from .experiment import Experiment
from .task import Task
from . import constants

__all__ = ['Experiment', 'Task', 'constants']

