import json
import os
import datetime
import logging
from typing import List, Dict, Any, Optional, Tuple, Union


import src.experiment.constants as constants
from src.prompt_processor.llm_processor import LLMProcessor
from src.prompt_processor.non_llm_processor import NonLLMProcessor
from src.llm_util.llm_model import LLMModel
from src.clas_jailbreaking_evaluation.score_calculator import ScoreCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Experiment:
    """Experiment class for running prompt processing experiments."""

    def __init__(self):
        """
        Initialize the experiment.

        Args:
            name: Name of the experiment
        """
        self.score_calculator = ScoreCalculator()
        self.results = {}
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        logger.info(f"Initialized experiment")

    def do_LLM_experiment(self, models: List[LLMModel] = constants.DEFAULT_LLM_MODELS) -> Dict[str, Any]:
        """
        Run an LLM experiment for each model and save results.
        
        Args:
            models: List of LLM models to test
            
        Returns:
            Dictionary with experiment results
        """
        
        # Run experiments for each model
        all_results = []
        
        for model in models:
            # Create a processor for this model
            processor = LLMProcessor(
                model=model,
                input_file=constants.DEFAULT_INPUT_FILE,
                output_file=constants.DEFAULT_OUTPUT_FILE
            )
            
            # Process prompts with this model
            processed_prompts = processor.process_and_save()
            
            # Calculate jailbreak and stealthiness scores if jailbreak evaluation is available
            scores = None
            
            if self.jailbreak_evaluation_available and self.score_calculator and processed_prompts:
                logger.info(f"Evaluating with model: {self.score_calculator.model_id}")
                try:
                    # Use the calculator's built-in scoring method and store results directly
                    scores = self.score_calculator.calculate_all_scores(
                        input_prompts_path=constants.DEFAULT_INPUT_FILE,
                        output_prompts_path=constants.DEFAULT_OUTPUT_FILE
                    )
                    logger.info(f"Scores: {scores}")
                except Exception as e:
                    logger.error(f"Error evaluating: {str(e)}")
            
            # Create a result record
            result = {
                "model": model.value,
                "prompts_processed": len(processed_prompts),
                "jailbreak_evaluation_available": self.jailbreak_evaluation_available,
                "evaluation_model": self.score_calculator.model_id if self.score_calculator else None
            }
            
            # Add scores if available
            if scores:
                result.update(scores)
            
            # Add to results
            all_results.append(result)
        
        # Store results in memory for potential future use
        combined_results = {
            "results": all_results,
            "jailbreak_evaluation_available": self.jailbreak_evaluation_available
        }
        self.results[constants.KEY_LLM_EXPERIMENT] = combined_results
        
        # Save each result as a separate JSON line
        save_path = constants.LLM_RESULTS_FILE
        # Open in write mode to overwrite previous results
        with open(save_path, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
        
        logger.info(f"LLM experiment completed and results saved to {save_path}")
        return combined_results

    def do_non_llm_experiment_and_save(self, non_llm_parameters_list: List[Dict[str, Any]] = constants.DEFAULT_NON_LLM_PARAMETERS_LIST) -> Dict[str, Any]:
        """
        Run the non-LLM experiment with specified modes and parts and save results.
        
        Args:
            non_llm_parameters_list: List of dictionaries containing mode and parts configurations to test
            
        Returns:
            Dictionary with experiment results
        """
        # Run experiments for each configuration in the list
        all_results = []
        
        for non_llm_parameters in non_llm_parameters_list:
            # Create a processor for this configuration
            processor = NonLLMProcessor(
                input_file=constants.DEFAULT_INPUT_FILE,
                output_file=constants.DEFAULT_OUTPUT_FILE
            )
            
            # Process prompts with this configuration
            processed_prompts = processor.process_and_save(non_llm_parameters=non_llm_parameters)
            
            # Calculate scores using the score calculator if available
            scores = None
            
            if processed_prompts:
                logger.info(f"Evaluating with model: {self.score_calculator.model_id}")
                try:
                    # Use the calculator's built-in scoring method and store results directly
                    scores = self.score_calculator.calculate_all_scores(
                        input_prompts_path=constants.DEFAULT_INPUT_FILE,
                        output_prompts_path=constants.DEFAULT_OUTPUT_FILE
                    )
                    logger.info(f"Scores: {scores}")
                except Exception as e:
                    logger.error(f"Error evaluating: {str(e)}")
            
            # Create a result record with detailed information
            result = {
                "prompts_processed": len(processed_prompts),
                "evaluation_model": self.score_calculator.model_id if self.score_calculator else None
            }
            
            # Add the whole non_llm_parameters to the result
            result.update(non_llm_parameters)
            
            # Add scores if available
            if scores:
                result.update(scores)
            
            # Add to results
            all_results.append(result)
            
            # Print detailed information about this result
            logger.info(f"Completed processing for parts: {non_llm_parameters}")
            logger.info(f"  Prompts processed: {len(processed_prompts)}")
            if scores:
                logger.info(f"  Final score: {scores.get('final_score', 0.0):.4f}")
        
        # Store results in memory
        combined_results = {
            "results": all_results,
        }
        self.results[constants.KEY_NON_LLM_EXPERIMENT] = combined_results
        
        # Save each result as a separate JSON line
        save_path = constants.NON_LLM_RESULTS_FILE
        # Open in write mode to overwrite previous results
        with open(save_path, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
        
        logger.info(f"Non-LLM experiment completed and results saved to {save_path}")
        logger.info(f"Results include {len(all_results)} configurations")
        
        return combined_results
  


