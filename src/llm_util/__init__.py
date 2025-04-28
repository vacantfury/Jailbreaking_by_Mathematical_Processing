"""
LLM utility module for working with various language models.
"""
from typing import Dict, Any

# Define version information
__version__ = "1.0.0"

# Import key components directly to maintain backward compatibility
from .llm_model import LLMModel
from .llm_service import LLMService, llm_service
from .llm_client import (
    BaseLLMClient, 
    OpenAIClient, 
    OllamaClient, 
    AnthropicClient,
    LLMClientFactory,
    ClientError, 
    ConfigurationError, 
    APIError
)
from .constants import LLM_CONFIG

# Lazy imports to improve startup time when not all modules are needed
import importlib.util
import sys
from typing import Optional, Callable, TypeVar, Generic, Dict, Any

T = TypeVar('T')

class LazyLoader(Generic[T]):
    """Lazy loader for module imports to improve startup time."""
    
    def __init__(self, import_path: str):
        self.import_path = import_path
        self._module: Optional[T] = None
        
    def __call__(self) -> T:
        if self._module is None:
            self._module = importlib.import_module(self.import_path)
        return self._module

# Create lazy loaders for components that may have dependencies
_llm_service_loader = LazyLoader('.llm_service')
_llm_client_loader = LazyLoader('.llm_client')

# Define getters for lazy-loaded components
def get_llm_service():
    """Get the LLMService class and singleton instance."""
    module = _llm_service_loader()
    return module.LLMService, module.llm_service

def get_clients():
    """Get the LLM client classes."""
    module = _llm_client_loader()
    return {
        'BaseLLMClient': module.BaseLLMClient,
        'OpenAIClient': module.OpenAIClient,
        'OllamaClient': module.OllamaClient,
        'AnthropicClient': module.AnthropicClient,
        'LLMClientFactory': module.LLMClientFactory,
        'ClientError': module.ClientError,
        'ConfigurationError': module.ConfigurationError,
        'APIError': module.APIError
    }

# Function to import everything at once if needed
def import_all() -> Dict[str, Any]:
    """Import all components from the module at once."""
    return {
        # Models
        'LLMModel': LLMModel,
        
        # Service
        'LLMService': LLMService, 
        'llm_service': llm_service,
        
        # Clients
        'BaseLLMClient': BaseLLMClient, 
        'OpenAIClient': OpenAIClient, 
        'OllamaClient': OllamaClient, 
        'AnthropicClient': AnthropicClient,
        'LLMClientFactory': LLMClientFactory,
        
        # Errors
        'ClientError': ClientError, 
        'ConfigurationError': ConfigurationError, 
        'APIError': APIError,
        
        # Configs
        'LLM_CONFIG': LLM_CONFIG
    }

# Define what's exported in __all__
__all__ = [
    # Direct imports for backward compatibility
    'LLMModel', 'LLM_CONFIG', 'LLMService', 'llm_service',
    'BaseLLMClient', 'OpenAIClient', 'OllamaClient', 'AnthropicClient',
    'LLMClientFactory', 'ClientError', 'ConfigurationError', 'APIError',
    
    # Lazy-loaded imports - to be resolved at runtime
    'get_llm_service', 'get_clients', 'import_all',
    
    # Module information
    '__version__'
] 