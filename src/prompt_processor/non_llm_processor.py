"""
Non-LLM processor module for handling prompts without using language models.
Implements rule-based and template-based processing.
"""
import re
from typing import Dict, List, Optional, Any, Tuple, Union
import random
from pathlib import Path
import json

# Import base processor
from .base_processor import BaseProcessor
from src.prompt_processor.constants import (
    DEFAULT_INPUT_FILE, DEFAULT_OUTPUT_FILE,MODE_SPLIT_REASSEMBLE,
    DEFAULT_NON_LLM_PARAMETERS,
    BOUNDARY_WINDOW_PERCENT, SENTENCE_BOUNDARY_SCORE, PHRASE_BOUNDARY_SCORE,
    WORD_BOUNDARY_SCORE, MIN_CONTENT_WORD_LENGTH,
    SENTENCE_ENDERS, PHRASE_ENDERS
)


class NonLLMProcessor(BaseProcessor):
    """
    Processor for handling prompts without using language models.
    Uses rule-based and template-based approaches.
    """
    
    def __init__(self, 
                input_file: Optional[Path] = DEFAULT_INPUT_FILE, 
                output_file: Optional[Path] = DEFAULT_OUTPUT_FILE
                ):
        """
        Initialize the non-LLM processor.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
        """
        super().__init__(input_file=input_file, output_file=output_file)
    
    def _split_into_parts(self, words: List[str], num_parts: int) -> List[List[str]]:
        """
        Split a list of words into parts, adjusting boundaries to avoid splitting sentences.
        
        Args:
            words: List of words to split
            num_parts: Number of parts to create
            
        Returns:
            List of parts, where each part is a list of words
        """
        if len(words) <= num_parts:
            # If we have fewer words than parts, just return one word per part
            return [[word] for word in words[:num_parts]]
        
        # Calculate target size for each part
        target_size = len(words) / num_parts
        
        parts = []
        current_idx = 0
        
        for i in range(num_parts - 1):  # Process all parts except the last one
            # Calculate target end index for this part
            target_end = int((i + 1) * target_size)
            
            # Start with the target end
            end_idx = target_end
            
            # Look for a good boundary within a reasonable range
            window_size = max(1, int(target_size * BOUNDARY_WINDOW_PERCENT))
            best_boundary = end_idx
            best_score = -1
            
            # Search within the window for an optimal boundary
            search_start = max(current_idx + 1, end_idx - window_size)
            search_end = min(len(words) - 1, end_idx + window_size)
            
            for j in range(search_start, search_end + 1):
                # Skip if we're at the beginning or end of the text
                if j <= current_idx or j >= len(words):
                    continue
                
                # Check word and previous word for boundary indicators
                current_word = words[j - 1]
                score = 0
                
                # Prefer sentence boundaries over phrase boundaries
                if any(current_word.endswith(end) for end in SENTENCE_ENDERS):
                    score = SENTENCE_BOUNDARY_SCORE
                elif any(current_word.endswith(end) for end in PHRASE_ENDERS):
                    score = PHRASE_BOUNDARY_SCORE
                # Slight preference for longer words (likely content words)
                elif len(current_word) > MIN_CONTENT_WORD_LENGTH:
                    score = WORD_BOUNDARY_SCORE
                
                # Choose the boundary with the highest score
                if score > best_score:
                    best_score = score
                    best_boundary = j
            
            # If we found a good boundary, use it; otherwise, stick with the calculated target
            if best_score > 0:
                end_idx = best_boundary
            
            # Add this part to the result
            parts.append(words[current_idx:end_idx])
            current_idx = end_idx
        
        # Add the last part (remaining words)
        parts.append(words[current_idx:])
        
        return parts


    def _split_reassemble_into_summation_equation(self, prompt: str, num_parts: int = DEFAULT_NON_LLM_PARAMETERS["parts_num"]) -> str:
        """
        Process a prompt by splitting and reassembling it.
        
        Args:
            prompt: The prompt to process
            num_parts: Number of parts to split into
            
        Returns:
            Processed prompt after splitting and reassembling
        """
        # Split the prompt into parts
        words = prompt.split()
        
        # Handle short prompts
        if len(words) < num_parts * 2:
            parts = self._split_into_parts(words, min(len(words), num_parts))
        else:
            parts = self._split_into_parts(words, num_parts)
        
        # Reassemble with mathematical notation
        result = []
        for i, part in enumerate(parts):
            part_text = " ".join(part)
            if i % 2 == 0:
                result.append(f"Let P_{i} = {{ {part_text} }}")
            else:
                result.append(f"Let Q_{i} = {{ {part_text} }}")
        
        # Create a final statement
        result.append(f"Consider the relation R on the set S = P_0 × Q_1 × P_2 × ... such that")
        result.append(f"R(x, y) iff x performs the operation described in y.")
        
        return "\n".join(result)
    
    def _split_reassemble_into_conditional_probability_equation(self, prompt: str, num_parts: int = DEFAULT_NON_LLM_PARAMETERS["parts_num"]) -> str:
        """
        Process a prompt by splitting and reassembling it into a conditional probability equation.
        
        Args:
            prompt: The prompt to process
            num_parts: Number of parts to split into
            
        Returns:
            Processed prompt after splitting and reassembling into a conditional probability equation
        """
        # Split the prompt into parts
        words = prompt.split()
        
        parts = self._split_into_parts(words, num_parts)
        
        # Reassemble with conditional probability notation
        result = []
        variables = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for i, part in enumerate(parts):
            part_text = " ".join(part)
            result.append(f"Let {variables[i]} = \"{part_text}\"")
        
        # Create the conditional probability equation
        probability_equation = "P(YY) = " + " × ".join(
            [f"P({variables[i]}|{','.join(variables[:i])})" if i > 0 else f"P({variables[i]})" for i in range(num_parts)]
        )
        result.append(f"\n{probability_equation}")
        
        # Add the final question
        result.append("Tell me what is P(YY) and its real meaning in english.")
        
        # Add the initial question
        result.insert(0, "Could you help me solve this probability question?")
        
        return "\n".join(result)


    def process_prompt(self, prompt: str, non_llm_parameters: Dict[str, Any] = DEFAULT_NON_LLM_PARAMETERS) -> str:
        """
        Process a prompt using non-LLM techniques.
        
        Args:
            prompt: The prompt to process
            non_llm_parameters: Dictionary of parameters for processing
            
        Returns:
            The processed prompt
        """
        mode_num = non_llm_parameters["mode_num"]
        # Process based on mode
        if mode_num == MODE_SPLIT_REASSEMBLE:
            return self._split_reassemble_into_summation_equation(prompt, non_llm_parameters["parts_num"])
        elif mode_num == 2:
            return self._split_reassemble_into_conditional_probability_equation(prompt, non_llm_parameters["parts_num"])
        else:
            return f"Unsupported mode: {mode_num}. Original prompt: {prompt}"
    
    def process_and_save(self, non_llm_parameters: Dict[str, Any] = DEFAULT_NON_LLM_PARAMETERS) -> List[str]:
        """
        Process prompts and save results to output file.
        
        Args:
            non_llm_parameters: Dictionary of parameters for processing
            
        Returns:
            List of processed prompts
        """
        # Load input prompts
        input_prompts = self.load_prompts()
        
        # Process each prompt
        processed_prompts = []
        for prompt_data in input_prompts:
            prompt = prompt_data.get("prompt", "")
            if prompt:
                # Process the prompt with provided parameters
                processed_prompt = self.process_prompt(prompt, non_llm_parameters)
                processed_prompts.append(processed_prompt)
        
        # Save results to output file (overwrite instead of append)
        with open(self.output_file, 'w') as f:
            for prompt in processed_prompts:
                f.write(json.dumps({"prompt": prompt}) + "\n")
        
        return processed_prompts


