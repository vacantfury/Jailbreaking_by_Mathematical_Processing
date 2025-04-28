"""
Constants for experiment package.
"""
from pathlib import Path
import os
from src.llm_util.llm_model import LLMModel
import src.clas_jailbreaking_evaluation.constants as clas_constants
from src.clas_jailbreaking_evaluation.constants import AVAILABLE_EVALUATION_MODELS
from src.prompt_processor.constants import MODE_KEYS_TO_NAMES
#############################################
# GENERAL EXPERIMENT SETTINGS
#############################################

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "input_and_output"
OUTPUT_DIR = PROJECT_ROOT / "input_and_output"

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default files
DEFAULT_INPUT_FILE = INPUT_DIR / "input_prompts.jsonl"
DEFAULT_OUTPUT_FILE = OUTPUT_DIR / "output_prompts.jsonl"

# Evaluation notebook path
DEFAULT_EVALUATION_IPYNB_PATH = PROJECT_ROOT / "evaluation" / "evaluate_responses.ipynb"

# Experiment result files
LLM_RESULTS_FILE = OUTPUT_DIR / "llm_results.jsonl"
NON_LLM_RESULTS_FILE = OUTPUT_DIR / "non_llm_results.jsonl"



KEY_LLM_EXPERIMENT = "llm_experiment"
KEY_NON_LLM_EXPERIMENT = "non_llm_experiment"



#############################################
# NON-LLM EXPERIMENT SETTINGS
#############################################


DEFAULT_NON_LLM_PARAMETERS_LIST = [
    {
        "mode_num": 2,
        "parts_num": 3
    },
    {
        "mode_num": 2,
        "parts_num": 4
    },  
    {
        "mode_num": 2,
        "parts_num": 5
    },
    {
        "mode_num": 2,
        "parts_num": 6
    }
]

#############################################
# LLM EXPERIMENT SETTINGS
#############################################

# Default LLM models to test
DEFAULT_LLM_MODELS = [LLMModel.GPT_3_5_TURBO, LLMModel.MISTRAL]

