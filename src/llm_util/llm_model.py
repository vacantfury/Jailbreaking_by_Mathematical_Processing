"""
LLM model definitions with provider information.
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, ClassVar, List, Set, cast, Final, Tuple
from functools import lru_cache


class LLMModel(Enum):
    """Enum for LLM models with provider information."""
    
    # OpenAI models
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    
    # Ollama models
    LLAMA2 = "llama2"
    LLAMA3 = "llama3"
    MISTRAL = "mistral"
    CODELLAMA = "codellama"
    
    # Internal model grouping
    _OPENAI_MODELS: ClassVar[Set['LLMModel']] = set()
    _OLLAMA_MODELS: ClassVar[Set['LLMModel']] = set()
    _ALL_PROVIDERS: ClassVar[Dict[str, Set['LLMModel']]] = {}
    _MODEL_NAME_MAP: ClassVar[Dict[str, 'LLMModel']] = {}
    
    @property
    def provider(self) -> str:
        """
        Get the provider for this model.
        
        Returns:
            Provider name (openai, ollama, etc.)
        """
        # This method is used frequently, so we cache the provider info
        # during class initialization to avoid the lookup each time
        if self in self._OPENAI_MODELS:
            return "openai"
        elif self in self._OLLAMA_MODELS:
            return "ollama"
        else:
            return "unknown"
    
    @classmethod
    def _init_model_groups(cls) -> None:
        """
        Initialize model groupings for faster lookups.
        Called automatically when the class is first used.
        """
        if cls._OPENAI_MODELS or cls._OLLAMA_MODELS:
            return  # Already initialized
            
        # Initialize model groups
        cls._OPENAI_MODELS = {
            cls.GPT_3_5_TURBO, 
            cls.GPT_4,
            cls.GPT_4_TURBO
        }
        
        cls._OLLAMA_MODELS = {
            cls.LLAMA2, 
            cls.LLAMA3,
            cls.MISTRAL,
            cls.CODELLAMA
        }
        
        # Create provider map for faster lookups
        cls._ALL_PROVIDERS = {
            "openai": cls._OPENAI_MODELS,
            "ollama": cls._OLLAMA_MODELS
        }
        
        # Create model name map for faster lookups
        cls._MODEL_NAME_MAP = {model.value: model for model in cls}
    
    @classmethod
    @lru_cache(maxsize=100)  # Cache the results for performance
    def from_string(cls, model_name: str) -> 'LLMModel':
        """
        Get a model enum from a string name.
        
        Args:
            model_name: Model name string
            
        Returns:
            Corresponding LLMModel enum
            
        Raises:
            ValueError: If model name is not recognized
        """
        # Initialize model groups if not already done
        cls._init_model_groups()
        
        # Fast lookup in the model name map
        if model_name in cls._MODEL_NAME_MAP:
            return cls._MODEL_NAME_MAP[model_name]
            
        raise ValueError(f"Unknown model: {model_name}")
    
    @classmethod
    def get_all_models(cls) -> Dict[str, List['LLMModel']]:
        """
        Get all available models by provider.
        
        Returns:
            Dictionary mapping provider to list of models
        """
        # Initialize model groups if not already done
        cls._init_model_groups()
        
        # Return a copy of the provider map with lists instead of sets
        return {
            provider: list(models) 
            for provider, models in cls._ALL_PROVIDERS.items()
        }
    
    @classmethod
    def get_models_by_provider(cls, provider: str) -> List['LLMModel']:
        """
        Get all models for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of models for the provider
        """
        # Initialize model groups if not already done
        cls._init_model_groups()
        
        # Get models for the provider
        return list(cls._ALL_PROVIDERS.get(provider, set()))


# Initialize model groups when module is loaded
LLMModel._init_model_groups()


class ModelConfig:
    """Configuration settings for specific models."""
    
    # Model configuration defaults - stored as class variable for better reuse
    _DEFAULT_CONFIGS: Final[Dict[LLMModel, Dict[str, Any]]] = {
        LLMModel.GPT_3_5_TURBO: {
            "temperature": 0.7,
            "max_tokens": 800,
            "context_length": 4096
        },
        LLMModel.GPT_4: {
            "temperature": 0.7,
            "max_tokens": 1000,
            "context_length": 8192
        },
        LLMModel.GPT_4_TURBO: {
            "temperature": 0.7,
            "max_tokens": 1500,
            "context_length": 128000
        },
        LLMModel.LLAMA2: {
            "temperature": 0.7,
            "max_tokens": 500,
            "context_length": 4096
        },
        LLMModel.LLAMA3: {
            "temperature": 0.7,
            "max_tokens": 700,
            "context_length": 8192
        },
        LLMModel.MISTRAL: {
            "temperature": 0.7,
            "max_tokens": 600,
            "context_length": 8192
        },
        LLMModel.CODELLAMA: {
            "temperature": 0.4,  # Lower temperature for code
            "max_tokens": 800,
            "context_length": 16000
        }
    }
    
    # Default config for any model not explicitly defined
    _FALLBACK_CONFIG: Final[Dict[str, Any]] = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "context_length": 4096
    }
    
    @classmethod
    @lru_cache(maxsize=32)  # Cache the results for performance
    def get_default_config(cls, model: LLMModel) -> Dict[str, Any]:
        """
        Get default configuration for a model.
        
        Args:
            model: The LLM model
            
        Returns:
            Dictionary with default configuration
        """
        # Return the config or fallback to default
        return cls._DEFAULT_CONFIGS.get(model, cls._FALLBACK_CONFIG).copy()
    
    @classmethod
    def update_default_config(cls, model: LLMModel, config_updates: Dict[str, Any]) -> None:
        """
        Update the default configuration for a model.
        
        Args:
            model: The model to update config for
            config_updates: New configuration values
        """
        if model not in cls._DEFAULT_CONFIGS:
            # Create a new config by copying the fallback
            cls._DEFAULT_CONFIGS[model] = cls._FALLBACK_CONFIG.copy()
            
        # Update the config with new values
        cls._DEFAULT_CONFIGS[model].update(config_updates)
        
        # Clear the cache to ensure updated values are used
        if hasattr(cls.get_default_config, 'cache_clear'):
            cls.get_default_config.cache_clear() 