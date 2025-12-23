"""
Constants for data paths and files.
"""
from pathlib import Path
from typing import Final

# Data root directory (src/data)
DATA_DIR: Final[Path] = Path(__file__).parent
DATASETS_DIR: Final[Path] = DATA_DIR / "datasets"

# Experiment Data Directory
EXPERIMENT_DATA_DIR: Final[Path] = DATA_DIR / "experiment_data"

# =============================================================================
# Dataset Directories & Files
# =============================================================================

# 1. HarmBench Small
HARMBENCH_SMALL_DIR: Final[Path] = DATASETS_DIR / "HarmBench_small"
HARMBENCH_SMALL_DATASET_PATH: Final[Path] = HARMBENCH_SMALL_DIR / "harmbench_small_dataset.json"

# 2. HarmBench Large
HARMBENCH_LARGE_DIR: Final[Path] = DATASETS_DIR / "HarmBench_large"
HARMBENCH_LARGE_DATASET_PATH: Final[Path] = HARMBENCH_LARGE_DIR / "harmbench_large_dataset.json"

# 3. Ata Data (Pilot)
ATA_DATA_DIR: Final[Path] = DATASETS_DIR / "Ata_data"
ATA_HARMFUL_PROMPTS_FILE: Final[Path] = ATA_DATA_DIR / "harmful_prompts.json"

# =============================================================================
# Experiment Output Directories
# =============================================================================
# These folders will contain the output for each dataset's experiments

HARMBENCH_SMALL_EXPERIMENT_DIR: Final[Path] = EXPERIMENT_DATA_DIR / "harmbench_small_dataset"
HARMBENCH_LARGE_EXPERIMENT_DIR: Final[Path] = EXPERIMENT_DATA_DIR / "harmbench_large_dataset"
ATA_EXPERIMENT_DIR: Final[Path] = EXPERIMENT_DATA_DIR / "Ata_data"


# Ensure directories exist
EXPERIMENT_DATA_DIR.mkdir(exist_ok=True, parents=True)
HARMBENCH_SMALL_EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)
HARMBENCH_LARGE_EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)
ATA_EXPERIMENT_DIR.mkdir(exist_ok=True, parents=True)
