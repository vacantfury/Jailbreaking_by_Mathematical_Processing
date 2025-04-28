"""
Base processor module for handling prompt processing.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Import constants
from .constants import DEFAULT_INPUT_FILE, DEFAULT_OUTPUT_FILE


class BaseProcessor:
    """Base class for all prompt processors."""
    
    def __init__(self, input_file: Optional[Union[str, Path]] = None, output_file: Optional[Union[str, Path]] = None):
        """
        Initialize the processor.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
        """
        # Set default paths if not provided
        self.input_file = Path(input_file) if input_file else DEFAULT_INPUT_FILE
        self.output_file = Path(output_file) if output_file else DEFAULT_OUTPUT_FILE
        
        # Ensure the output directory exists
        os.makedirs(self.output_file.parent, exist_ok=True)
    
    def process_prompt(self, prompt: str) -> str:
        """
        Process a single prompt.
        
        Args:
            prompt: The prompt to process
            
        Returns:
            The processed prompt
        """
        # Default implementation - to be overridden by subclasses
        return f"Processed: {prompt}"
    
    def process_and_save(self) -> List[str]:
        """
        Process all prompts from the input file and save to the output file.
        
        Returns:
            List of processed prompts
        """
        # Load input prompts
        input_prompts = self.load_prompts()
        
        # Process each prompt
        processed_prompts = []
        for prompt_data in input_prompts:
            prompt = prompt_data.get("prompt", "")
            
            # Process the prompt
            response = self.process_prompt(prompt)
            processed_prompts.append(response)
            
            # Save the processed prompt
            self.save_prompt({
                "prompt": prompt,
                "response": response,
                "processor": self.__class__.__name__
            })
        
        return processed_prompts
    
    def load_prompts(self) -> List[Dict[str, Any]]:
        """
        Load prompts from the input file.
        
        Returns:
            List of prompt dictionaries
        """
        prompts = []
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        prompts.append(json.loads(line))
        except Exception as e:
            print(f"Error loading prompts from {self.input_file}: {str(e)}")
        
        return prompts
    
    def save_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """
        Save a processed prompt to the output file.
        
        Args:
            prompt_data: The prompt data to save
        """
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(prompt_data) + '\n')
        except Exception as e:
            print(f"Error saving prompt to {self.output_file}: {str(e)}") 