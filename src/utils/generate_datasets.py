import pandas as pd
import json
import sys
from pathlib import Path

# Fix python path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.constants import (
    HARMBENCH_SMALL_DATASET_PATH,
    HARMBENCH_LARGE_DATASET_PATH
)

# Source Paths (HarmBench Data)
# Corrected path based on find_by_name: src/data/original_datasets/HarmBench_data/behavior_datasets/
HARMBENCH_VAL_CSV = Path("src/data/original_datasets/HarmBench_data/behavior_datasets/harmbench_behaviors_text_val.csv")
HARMBENCH_ALL_CSV = Path("src/data/original_datasets/HarmBench_data/behavior_datasets/harmbench_behaviors_text_all.csv")

def ensure_dir_exists(file_path: Path):
    """Ensure directory for the file exists."""
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {file_path.parent}")

def save_json(data: list, output_path: Path):
    """Save data list to JSON file."""
    ensure_dir_exists(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved {len(data)} items to {output_path}")

def generate_small_dataset():
    """Generates the 'Small' dataset (stratified subset of ~120 items)."""
    print("\n--- Generating Small Dataset ---")
    
    # Use ALL dataset for better variety if VAL is too small
    # But sticking to VAL for now if user wants validation set, 
    # checking source availability... 
    # Actually, VAL only has 317 items. 20 per category * 6 = 120. It fits.
    source_csv = HARMBENCH_VAL_CSV
    if not source_csv.exists():
        # Fallback to ALL if VAL missing
        source_csv = HARMBENCH_ALL_CSV
    
    if not source_csv.exists():
        print(f"Error: Source file not found: {source_csv}")
        return

    df = pd.read_csv(source_csv)
    
    # Target Categories (Excluding Copyright)
    categories = [
        "chemical_biological",
        "cybercrime_intrusion",
        "illegal",
        "misinformation_disinformation",
        "harmful",
        "harassment_bullying"
    ]
    
    # Stratified Sampling (20 per category -> 120 total)
    selected_rows = []
    for cat in categories:
        cat_df = df[df["SemanticCategory"] == cat]
        sample = cat_df.head(20) # Selected 20 items
        selected_rows.append(sample)
        print(f"Category '{cat}': Selected {len(sample)} items")
        
    final_df = pd.concat(selected_rows)
    
    # Convert to List of Dicts (Standard Format)
    data = final_df.to_dict(orient="records")
    
    # Basic normalization (ensure 'id' field exists if needed, though 'BehaviorID' is standard)
    # Mapping 'BehaviorID' -> 'id' and 'Behavior' -> 'prompt' if used by loader
    normalized_data = []
    for item in data:
        normalized_data.append({
            "id": item["BehaviorID"],
            "prompt": item["Behavior"],
            "category": item["SemanticCategory"],
            "source": "HarmBench"
        })

    save_json(normalized_data, HARMBENCH_SMALL_DATASET_PATH)

def generate_large_dataset():
    """Generates the 'Large' dataset (All non-copyright items)."""
    print("\n--- Generating Large Dataset ---")
    if not HARMBENCH_ALL_CSV.exists():
        print(f"Error: Source file not found: {HARMBENCH_ALL_CSV}")
        return

    df = pd.read_csv(HARMBENCH_ALL_CSV)
    
    # Filter out Copyright
    filtered_df = df[df["SemanticCategory"] != "copyright"]
    print(f"Filtered out 'copyright'. Remaining items: {len(filtered_df)}")
    
    # Convert to List of Dicts
    data = filtered_df.to_dict(orient="records")
    
    normalized_data = []
    for item in data:
        normalized_data.append({
            "id": item["BehaviorID"],
            "prompt": item["Behavior"],
            "category": item["SemanticCategory"],
            "source": "HarmBench"
        })
        
    save_json(normalized_data, HARMBENCH_LARGE_DATASET_PATH)

if __name__ == "__main__":
    generate_small_dataset()
    generate_large_dataset()
