"""
LLM processor for processing prompts through language models.
"""
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.llm_util.llm_service import llm_service
from src.llm_util.llm_model import LLMModel
from src.prompt_processor.constants import PROMPT_TEMPLATE, PREVIOUS_EXAMPLE
from src.prompt_processor.base_processor import BaseProcessor

# Set up logging
logger = logging.getLogger(__name__)

class LLMProcessor(BaseProcessor):
    """Processor class for handling prompt processing through LLMs."""
    
    def __init__(self, 
                model: LLMModel = LLMModel.GPT_3_5_TURBO,
                system_message: str = "You are a helpful assistant.",
                temperature: float = 0.7,
                max_tokens: int = 2000,
                input_file: Optional[Path] = None,
                output_file: Optional[Path] = None,
                use_example: bool = False):
        """
        Initialize the LLM processor.
        
        Args:
            model: The LLM model to use
            system_message: System message for the LLM
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            input_file: Input file path for processing multiple prompts
            output_file: Output file path for saving processed prompts
            use_example: Whether to use an example in the prompt
        """
        super().__init__(input_file=input_file, output_file=output_file)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_example = use_example
    
    def process_prompt(self, prompt: str) -> str:
        """
        Process a prompt using the LLM service.
        
        Args:
            prompt: The prompt to process
            
        Returns:
            The processed prompt
        """
        # Format prompt with example if needed
        if self.use_example:
            formatted_prompt = self._format_with_example(prompt)
        else:
            formatted_prompt = prompt
            
        # Invoke LLM through the service
        return llm_service.invoke_llm(
            prompt=formatted_prompt,
            model=self.model,
            system_message=self.system_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
    
    def process_and_save(self) -> List[Dict[str, str]]:
        """
        Process prompts and save results to output file.
        
        Returns:
            List of dictionaries containing original prompts, processed prompts, and responses
        """
        # Process prompts
        processed_prompts = self.process_prompt()
        
        # Save results to output file (overwrite instead of append)
        with open(self.output_file, 'w') as f:
            for prompt_data in processed_prompts:
                f.write(json.dumps(prompt_data) + "\n")
        
        return processed_prompts
    
    def _format_with_example(self, prompt: str) -> str:
        """
        Format a prompt with a system message and example.
        
        Args:
            prompt: The user prompt
            
        Returns:
            Formatted prompt with example
        """
        return PROMPT_TEMPLATE.format(
            system_message=self.system_message,
            previous_example=PREVIOUS_EXAMPLE,
            user_prompt=prompt
        )


# Simple usage example
if __name__ == "__main__":
    # Create a LLM processor for GPT
    processor = LLMProcessor(
        model=LLMModel.GPT_3_5_TURBO,
        output_file="llm_output.jsonl"
    )
    
    # Test with a single prompt
    test_prompt = "Explain the concept of recursion in simple terms."
    response = processor.process_prompt(test_prompt)
    
    print("=== LLM Response ===")
    print(response) 