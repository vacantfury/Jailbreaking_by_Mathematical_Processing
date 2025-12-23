# Datasets Documentation

This directory contains datasets used for jailbreak experiments, primarily derived from **HarmBench**.

## Available Datasets

### 1. HarmBench Small (`harmbench_small`)
- **Path**: `src/data/harmbench_small_dataset/harmbench_small_dataset.json`
- **Purpose**: Rapid testing, debugging, and verification.
- **Size**: ~120 items.
- **Composition**: Stratified sample of 20 distinct behaviors from each of the 6 high-risk categories (Chemical, Cybercrime, Illegal, Misinformation, Harmful, Harassment).
- **Source**: `harmbench_behaviors_text_val.csv` (Validation split).

### 2. HarmBench Large (`harmbench_large`)
- **Path**: `src/data/harmbench_large_dataset/harmbench_large_dataset.json`
- **Purpose**: Full-scale paper experiments and rigorous evaluation.
- **Size**: ~300 items.
- **Composition**: All available behaviors in the dataset excluding the "Copyright" category (which is not relevant for safety jailbreaking).
- **Source**: `harmbench_behaviors_text_all.csv` (Full dataset).

## Usage in Experiments

You can select these datasets in your experiment code by using their alias:

```python
# In experiment_test.py
experiments.add(
    name="My_Task",
    processing_strategy=ProcessorType.LLM_SET_THEORY,
    target_model=LLMModel.GPT_4O,
    input_data_path="harmbench_small"  # Use alias here
)
```

## Generation
These datasets are generated using `src/utils/generate_datasets.py`.
```bash
python src/utils/generate_datasets.py
```
