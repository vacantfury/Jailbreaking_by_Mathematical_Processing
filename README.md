# LLM Experiment Framework

A comprehensive framework for running experiments with Large Language Models (LLMs) and evaluating their performance.

## 🌟 Features

- **Unified Configuration**: Centralized configuration for all components
- **Multiple LLM Support**: Integration with OpenAI GPT and Ollama (Llama) models
- **Extensible Architecture**: Easy to add new models and processors
- **Experiment Management**: Track, save, and compare experiment results
- **Jailbreaking Analysis**: Tools for analyzing jailbreaking attack effectiveness
- **LangChain Integration**: Built on top of the LangChain framework

## 📋 Project Structure

```
.
├── config.py                      # Central configuration file
├── setup.py                       # Installation script
├── experiment/                    # Experiment framework
│   ├── __init__.py
│   ├── experiment_manager.py      # Core experiment management
│   └── constants.py               # Experiment-specific constants
├── prompt_processing/             # Prompt processing components
│   ├── __init__.py
│   ├── processor.py               # Unified processor interface
│   ├── base_processor.py          # Legacy base processor
│   ├── LLM_processor.py           # Legacy LLM processor
│   ├── gpt_processor.py           # GPT processor implementation
│   ├── llama_processor.py         # Llama processor implementation
│   ├── non_LLM_processor.py       # Non-LLM processor
│   └── constants.py               # Prompt-related constants
├── llm_util/                      # LLM utility functions
│   ├── __init__.py
│   ├── llm_client.py              # LLM client interface
│   └── llm_util_example.py        # Example usage
├── jailbreaking_attack_track/     # Jailbreaking attack analysis
│   ├── score_calculator.py        # Calculate attack scores
│   └── analyze_results.py         # Analyze attack results
└── input_and_output/              # Input and output data
    ├── input_prompts.jsonl        # Input prompts
    ├── output_prompts.jsonl       # Processed prompts
    └── experiment_results.jsonl   # Experiment results
```

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-experiment-framework.git
cd llm-experiment-framework

# Install the package and dependencies
pip install -e .
```

### Basic Usage

#### 1. Configure your API Keys

For OpenAI:
```
# Set environment variable
export OPENAI_API_KEY=your_key_here

# Or create a key file
echo "your_key_here" > prompt_processing/ChatGPT_key.txt
```

#### 2. Run a Simple Experiment

```python
from experiment.experiment_manager import ExperimentManager

# Create an experiment manager
manager = ExperimentManager("my_experiment")

# Run an experiment with GPT and Llama
result = manager.run_llm_experiment(
    llm_providers=['openai', 'ollama'],
    model_names={
        'openai': 'gpt-3.5-turbo',
        'ollama': 'llama2'
    },
    trials=1
)

print(f"Experiment completed! Results saved to {manager.output_file}")
```

#### 3. Analyze Jailbreaking Results

```bash
python -m jailbreaking_attack_track.analyze_results \
    --input path/to/results.jsonl \
    --output analysis_results.json \
    --verbose
```

## 📊 Architecture

The framework is built around these core concepts:

1. **LLM Clients**: Wrappers around LLM APIs (OpenAI, Ollama, etc.)
2. **Processors**: Components that take input prompts and produce responses
3. **Experiment Manager**: Coordinates running experiments with multiple processors
4. **Score Calculator**: Analyzes and compares results

## 🔧 Configuration

The `config.py` file contains all the central configuration for the project:

- File paths and default files
- LLM settings (models, parameters)
- System prompts and templates
- Experiment settings
- Jailbreaking attack settings

## 🧩 Extending the Framework

### Adding a New LLM Provider

1. Create a new client in `llm_util/llm_client.py`:
```python
class NewProviderClient(BaseLLMClient):
    def _setup(self):
        # Setup code
        
    def query(self, prompt, system_message=None):
        # Implementation
```

2. Add to the factory:
```python
if provider == "new_provider":
    return NewProviderClient(**kwargs)
```

3. Create a processor in `prompt_processing/processor.py`:
```python
class NewProviderProcessor(LLMProcessor):
    def __init__(self, model_name=None, ...):
        super().__init__(
            provider="new_provider",
            model_name=model_name,
            ...
        )
```

## 📚 References

- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Ollama Documentation](https://ollama.ai/docs) 