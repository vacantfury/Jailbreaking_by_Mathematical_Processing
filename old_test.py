import os
import json
import sys
import logging
from pathlib import Path

# Configure logging without timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.experiment.experiment import Experiment
from src.llm_util.llm_model import LLMModel
from src.experiment import constants
from src.clas_jailbreaking_evaluation.score_calculator import ScoreCalculator


def test_score_calculator():
    score_calculator = ScoreCalculator()
    print(score_calculator.calculate_all_scores())
     
    
def test_llm_experiment():
    """Test the LLM experiment functionality."""
    print("Testing LLM experiment...")
    experiment = Experiment(name="test_experiment")
    
    # Run with just one model to keep test fast
    results = experiment.do_LLM_experiment(
        models=[LLMModel.GPT_3_5_TURBO]
    )
    
    # Check the output file exists and has content
    try:
        print("\nVerifying LLM results file:")
        all_results = []
        models = set()
        jailbreak_evaluation_available = False
        
        with open(constants.LLM_RESULTS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    result = json.loads(line)
                    all_results.append(result)
                    
                    # Extract metadata from the individual results
                    if "model" in result:
                        models.add(result["model"])
                    if "jailbreak_evaluation_available" in result:
                        jailbreak_evaluation_available = jailbreak_evaluation_available or result["jailbreak_evaluation_available"]
        
        if all_results:
            print(f"File contains {len(all_results)} result entries")
            print(f"Models used: {list(models)}")
            print(f"Jailbreak evaluation available: {jailbreak_evaluation_available}")
        else:
            print("Results file is empty")
    except Exception as e:
        print(f"Error reading results file: {str(e)}")
    
    print("LLM experiment test completed.")
    return results

def test_non_llm_experiment():
    """Test the non-LLM experiment functionality."""
    print("Testing non-LLM experiment...")
    
    # Create experiment with a clear name
    experiment = Experiment()
   
    results = experiment.do_non_llm_experiment_and_save()


def test(mode: str):
    """Run a test based on the specified mode.
    
    Args:
        mode: Type of test to run - "llm", "non_llm", or "scores"
    """
    
    if mode.lower() == "llm":
        result = test_llm_experiment()
        print(f"Test completed with {len(result.get('results', []))} results")
    elif mode.lower() == "non_llm":
        test_non_llm_experiment()
    elif mode.lower() == "scores":
        test_score_calculator()
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'llm', 'non_llm', or 'scores'")

if __name__ == "__main__":
    test("non_llm")