# Experiment Package

A framework for running, tracking, and evaluating LLM experiments.

## Overview

The experiment package provides tools to systematically run experiments with different LLM models, track their performance, and evaluate their results. It's designed to work with the `llm_util` and `prompt_processing` packages to create a complete experiment pipeline.

## Components

### ExperimentManager

The main class that manages experiments:

- Runs experiments with multiple processor types
- Tracks experiment results including duration and scores
- Supports custom evaluation functions
- Handles saving and loading experiment results
- Provides simplified API for LLM experiments

### Constants

Package-specific constants including:

- Default input/output file paths
- Experiment configuration parameters
- Default experiment settings

## Usage

### Basic Usage

```python
from experiment import ExperimentManager
from llm_util import LLMModel

# Create an experiment manager
manager = ExperimentManager("test_experiment")

# Run an experiment with specific models
results = manager.run_llm_experiment(
    models=[LLMModel.GPT_3_5_TURBO, LLMModel.LLAMA2],
    trials=3
)

# Print results
print(f"Experiment results: {results}")
```

### With Custom Evaluation

```python
from experiment import ExperimentManager
from llm_util import LLMModel

# Create an experiment manager
manager = ExperimentManager("eval_experiment")

# Define a custom evaluator
def accuracy_evaluator(prompt, response):
    # Check if response contains key information from the prompt
    return 1.0 if any(keyword in response for keyword in prompt.split()) else 0.0

# Run experiment with custom evaluation
results = manager.run_llm_experiment(
    models=[LLMModel.GPT_4, LLMModel.LLAMA3],
    evaluator=accuracy_evaluator,
    trials=5
)

# Get average scores by model
for model_result in results['results']:
    print(f"{model_result['model']}: {model_result.get('avg_score', 'N/A')}")
```

### Custom Processors

```python
from experiment import ExperimentManager
from prompt_processing import PromptProcessor, LLMProcessor
from llm_util import LLMModel

# Create custom processors
processors = [
    LLMProcessor(model=LLMModel.GPT_3_5_TURBO),
    LLMProcessor(model=LLMModel.LLAMA2, system_message="You are a math expert"),
    # Add your custom processor that extends PromptProcessor
    MyCustomProcessor()
]

# Create an experiment manager
manager = ExperimentManager("custom_experiment")

# Run experiment with custom processors
results = manager.run_experiment(processors, trials=2)
```

## Integration with Other Packages

The experiment package is designed to work seamlessly with:

- `llm_util`: Provides the LLM models and service for API communication
- `prompt_processing`: Handles prompt formatting and processing

## Extending the Framework

To extend the framework:

1. Create custom processors by extending `PromptProcessor`
2. Implement custom evaluator functions for specific metrics
3. Add new experiment configurations in `constants.py`

## Performance Considerations

- For large experiments, consider increasing the number of trials for statistical significance
- Use appropriate model parameters (temperature, max_tokens) to balance quality and speed
- For experiments with many prompts, consider batching or parallel processing 