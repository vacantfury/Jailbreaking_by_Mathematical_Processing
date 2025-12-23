"""
Constants for experiment package.
"""
from pathlib import Path
from typing import Final

from src.data.constants import (
    EXPERIMENT_DATA_DIR,
    ATA_HARMFUL_PROMPTS_FILE,
    HARMBENCH_SMALL_DATASET_PATH,
    HARMBENCH_LARGE_DATASET_PATH,
    HARMBENCH_SMALL_EXPERIMENT_DIR,
    HARMBENCH_LARGE_EXPERIMENT_DIR,
    ATA_EXPERIMENT_DIR
)
from src.llm_utils import LLMModel

# Re-export for backward compatibility if needed, or just update usages
HARMFUL_PROMPTS_FILE = ATA_HARMFUL_PROMPTS_FILE

# Dataset mapping
DATASET_NAME_TO_PATH = {
    "harmbench_small": HARMBENCH_SMALL_DATASET_PATH,
    "harmbench_large": HARMBENCH_LARGE_DATASET_PATH,
    "harm_bench_small": HARMBENCH_SMALL_DATASET_PATH, # Alias
    "harm_bench_large": HARMBENCH_LARGE_DATASET_PATH,  # Alias
    "ata_harmful": ATA_HARMFUL_PROMPTS_FILE
}

DATASET_NAME_TO_EXPERIMENT_PATH = {
    "harmbench_small": HARMBENCH_SMALL_EXPERIMENT_DIR,
    "harmbench_large": HARMBENCH_LARGE_EXPERIMENT_DIR,
    "harm_bench_small": HARMBENCH_SMALL_EXPERIMENT_DIR, # Alias
    "harm_bench_large": HARMBENCH_LARGE_EXPERIMENT_DIR,  # Alias
    "ata_harmful": ATA_EXPERIMENT_DIR
}

# =============================================================================
# Directory Paths
# =============================================================================

# Import base experiment data directory from data.constants
# All task folders will be created under this directory
TASK_DATA_DIR: Final[Path] = EXPERIMENT_DATA_DIR

# Ensure directories exist
TASK_DATA_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# Experiment Settings
# =============================================================================

# Default task name
DEFAULT_TASK_NAME: Final[str] = "jailbreaking_by_maths_task"

# Default temperature settings
DEFAULT_TARGET_TEMPERATURE: Final[float] = 1.0  # Higher for creative jailbreak attempts
DEFAULT_EVALUATOR_TEMPERATURE: Final[float] = 0.0  # Deterministic for consistent evaluation


# =============================================================================
# Batch Processing Settings
# =============================================================================

# Maximum number of prompts to process in one batch
MAX_PROMPTS_PER_BATCH: Final[int] = 100

# Maximum number of evaluations in one batch
MAX_EVALUATIONS_PER_BATCH: Final[int] = 50
