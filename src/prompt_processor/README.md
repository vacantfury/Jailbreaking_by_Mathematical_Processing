# Prompt Processing

A framework for processing prompts using various techniques including LLMs and rule-based approaches.

## Overview

The prompt processing package provides specialized processors for handling prompts with different techniques. Each processor type is implemented in its own file for clarity and maintainability.

## Components

### Base Processor (`base_processor.py`)

Base class for all prompt processors with common functionality:

- Loading prompts from input files
- Saving processed prompts to output files
- Managing file I/O and error handling

### LLM Processor (`llm_processor.py`)

Processor for handling prompts using Language Models:

- Unified interface for all LLM models
- Direct integration with `llm_service` for API communication
- Support for custom system messages and examples
- Flexible prompt formatting options

### Non-LLM Processor (`non_llm_processor.py`)

Processor for handling prompts without using language models:

- Rule-based and template-based approaches
- Multiple processing modes (templates, split-reassemble, keywords)
- Highly customizable for different use cases
- Deterministic output (unlike LLMs)

## File Structure

```
prompt_processing/
├── __init__.py             # Package exports
├── base_processor.py       # Base PromptProcessor class
├── llm_processor.py        # LLMProcessor implementation
├── non_llm_processor.py    # NonLLMProcessor implementation
├── processor.py            # Legacy imports for backward compatibility
└── constants.py            # Shared constants and configurations
```

## Usage

### LLM Processor

```python
from prompt_processing.llm_processor import LLMProcessor
from llm_util.llm_model import LLMModel

# Create a processor for a specific model
processor = LLMProcessor(
    model=LLMModel.GPT_3_5_TURBO,
    input_file="input.jsonl",
    output_file="output.jsonl"
)

# Process a single prompt
response = processor.process_prompt("Explain the concept of recursion.")
print(response)

# Process all prompts from the input file and save results
processor.process_and_save()
```

### Non-LLM Processor

```python
from prompt_processing.non_llm_processor import NonLLMProcessor

# Create a processor with template mode
template_processor = NonLLMProcessor(
    mode=1,  # Template mode
    input_file="input.jsonl",
    output_file="template_output.jsonl"
)

# Create a processor with split-reassemble mode
split_processor = NonLLMProcessor(
    mode=2,  # Split-reassemble mode
    num_parts=3,
    input_file="input.jsonl",
    output_file="split_output.jsonl"
)

# Process prompts with different modes
template_response = template_processor.process_prompt("How to create a secure password")
split_response = split_processor.process_prompt("How to create a secure password")

# Process with a specific mode and save
template_processor.process_and_save()
```

## Integration with Other Packages

The prompt processing package is designed to work seamlessly with:

- `llm_util`: Provides the LLM models and service for API communication
- `experiment`: Handles running experiments with multiple processors

## Design Philosophy

Each processor type has been isolated into its own file to:

1. Improve readability and maintainability
2. Allow for easier extension and customization
3. Enable independent development and testing
4. Support clearer separation of concerns

The package maintains a clean, hierarchical design with a base class (`PromptProcessor`) and specialized implementations for different processing strategies. 