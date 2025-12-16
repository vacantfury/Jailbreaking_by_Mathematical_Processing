"""
Factory for creating LLM service instances.
"""
from typing import Dict, Type

from .llm_model import LLMModel, Provider
from .base_llm_service import BaseLLMService

# Import concrete service implementations
from .openai_service import OpenAIService
from .claude_service import ClaudeService
from .google_service import GoogleService
from .local_lm_service import LocalLMService


class LLMServiceFactory:
    """Factory for creating LLM service instances based on model provider."""
    
    # Registry mapping providers to their service implementations
    _PROVIDER_REGISTRY: Dict[Provider, Type[BaseLLMService]] = {
        Provider.OPENAI: OpenAIService,
        Provider.ANTHROPIC: ClaudeService,
        Provider.GOOGLE: GoogleService,
        Provider.TRANSFORMERS: LocalLMService
    }
    
    
    @classmethod
    def get_registered_providers(cls) -> list[Provider]:
        """
        Get list of currently registered providers.
        
        Returns:
            List of provider enums that have registered services
        """
        return list(cls._PROVIDER_REGISTRY.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: Provider) -> bool:
        """
        Check if a provider is currently supported.
        
        Args:
            provider: The provider to check
        
        Returns:
            True if provider has a registered service, False otherwise
        """
        return provider in cls._PROVIDER_REGISTRY

    @classmethod
    def register_provider(cls, provider: Provider, service_class: Type[BaseLLMService]) -> None:
        """
        Register a service class for a provider.
        
        This allows dynamic registration of new providers without modifying the factory.
        
        Args:
            provider: The provider enum
            service_class: The service class to handle this provider
        """
        cls._PROVIDER_REGISTRY[provider] = service_class

    @classmethod
    def create(cls, model: LLMModel, **kwargs) -> BaseLLMService:
        """
        Create an LLM service instance for the given model.
        
        Args:
            model: The LLM model to create a service for
            **kwargs: Additional arguments passed to the service constructor
                Common kwargs:
                - temperature (float): Sampling temperature
                - max_tokens (int): Maximum tokens to generate
                - api_key (str): API key for API-based services
        
        Returns:
            Instance of the appropriate service implementation
        
        Raises:
            ValueError: If no service is registered for the model's provider
        
        Example:
            service = LLMServiceFactory.create(
                LLMModel.GPT_4,
                temperature=0.7,
                max_tokens=1000
            )
            
            prompts = [("id1", "What is AI?")]
            results = service.batch_generate(prompts)
        """
        service_class = cls._PROVIDER_REGISTRY.get(model.provider)
        
        if service_class is None:
            raise ValueError(
                f"No service registered for provider: {model.provider}. "
                f"Available providers: {list(cls._PROVIDER_REGISTRY.keys())}"
            )
        
        return service_class(model, **kwargs)