"""
Constants for data paths and files.
"""
from pathlib import Path

# Data directory paths
DATA_ROOT = Path(__file__).parent
PROMPTS_DIR = DATA_ROOT / "datasets"
EXPERIMENT_DATA_DIR = DATA_ROOT / "experiment_data"

# Prompt data files
HARMFUL_PROMPTS_FILE = PROMPTS_DIR / "harmful_prompts.json"

# Create directories if they don't exist
EXPERIMENT_DATA_DIR.mkdir(exist_ok=True)