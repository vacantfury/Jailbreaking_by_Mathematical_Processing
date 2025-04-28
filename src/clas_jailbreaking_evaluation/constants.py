import os
import src.env_config
# Weight factors for calculating the final score
JAILBREAK_WEIGHT = 0.84
STEALTHINESS_WEIGHT = 0.16

# Default path for input/output files
DEFAULT_INPUT_PROMPTS_PATH = "src/input_and_output/input_prompts.jsonl"
DEFAULT_OUTPUT_PROMPTS_PATH = "src/input_and_output/output_prompts.jsonl"
DEFAULT_TEST_RESULTS_PATH = "src/input_and_output/test_results.jsonl"

# models
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
AVAILABLE_EVALUATION_MODELS = ["google/gemma-2b-it", "meta-llama/Meta-Llama-3-8B-Instruct", "microsoft/deberta-v3-base"]
DEFAULT_MODEL = "google/gemma-2b-it"



