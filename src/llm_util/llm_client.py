"""
LLM client module with extensible architecture for different LLM providers.
"""
import os
import abc
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, cast
from pathlib import Path
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_client')

# LangChain imports
try:
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.chat_models import ChatOpenAI
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_community.llms import Ollama
    LANGCHAIN_AVAILABLE = True
except ImportError:
    logger.warning("LangChain not available. Some functionality will be limited.")
    LANGCHAIN_AVAILABLE = False

# Import local constants and models
from .constants import LLM_CONFIG
from .llm_model import LLMModel, ModelConfig

# Type for the client class
T = TypeVar('T', bound='BaseLLMClient')


class ClientError(Exception):
    """Base exception for client errors."""
    pass


class ConfigurationError(ClientError):
    """Exception for configuration errors."""
    pass


class APIError(ClientError):
    """Exception for API errors."""
    pass


class BaseLLMClient(abc.ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, 
                 model: Union[str, LLMModel], 
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 cache_size: int = 32):
        """
        Initialize the LLM client.
        
        Args:
            model: Model name or LLMModel enum
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            cache_size: Size of response cache
        """
        # Convert string model to enum if needed
        if isinstance(model, str):
            try:
                self.model = LLMModel.from_string(model)
            except ValueError:
                raise ConfigurationError(f"Unknown model: {model}")
        else:
            self.model = model
            
        # Get default config for this model
        default_config = ModelConfig.get_default_config(self.model)
        
        # Set parameters
        self.model_name = self.model.value
        self.temperature = temperature if temperature is not None else default_config.get("temperature", 0.7)
        self.max_tokens = max_tokens if max_tokens is not None else default_config.get("max_tokens", 800)
        self.cache_size = cache_size
        
        # Set up client
        try:
            self._setup()
            logger.info(f"Initialized {self.__class__.__name__} for model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {str(e)}")
            raise ConfigurationError(f"Client initialization failed: {str(e)}")
    
    @abc.abstractmethod
    def _setup(self) -> None:
        """Set up the client (to be implemented by subclasses)."""
        pass
    
    @abc.abstractmethod
    def _query_uncached(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Query the LLM with a prompt (uncached version).
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            The model's response
        """
        pass
    
    @lru_cache(maxsize=32)  # Default cache size, will be updated in __init__
    def _query_with_cache(self, prompt: str, system_message_hash: Optional[str] = None) -> str:
        """Cached version of query implementation."""
        # Convert the hash back to the system message or None
        system_message = None if system_message_hash is None else system_message_hash
        return self._query_uncached(prompt, system_message)
    
    def query(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Query the LLM with a prompt (with caching).
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            The model's response
            
        Raises:
            APIError: If there's an error with the API request
            ConfigurationError: If there's a configuration issue
            ClientError: For other client errors
        """
        try:
            # Create a hash of the system message for caching
            # None system_message remains None for the cache key
            system_message_hash = system_message
            
            # Update cache_size for the lru_cache decorator if needed
            if getattr(self._query_with_cache, 'cache_parameters', {}).get('maxsize') != self.cache_size:
                # Create a new cached function with the updated cache size
                self._query_with_cache = lru_cache(maxsize=self.cache_size)(self._query_uncached)
            
            return self._query_with_cache(prompt, system_message_hash)
        except Exception as e:
            if isinstance(e, (APIError, ConfigurationError)):
                raise
            logger.error(f"Error in query: {str(e)}")
            raise ClientError(f"Query failed: {str(e)}")
    
    async def aquery(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Asynchronously query the LLM.
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            The model's response
        """
        # Run synchronous query in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(prompt, system_message))
    
    async def abatch_query(self, prompts: List[str], system_message: Optional[str] = None) -> List[str]:
        """
        Process multiple prompts in batch asynchronously.
        
        Args:
            prompts: List of prompts to process
            system_message: Optional system message to use for all prompts
            
        Returns:
            List of responses
        """
        tasks = [self.aquery(prompt, system_message) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def batch_query(self, prompts: List[str], system_message: Optional[str] = None) -> List[str]:
        """
        Process multiple prompts in batch.
        
        Args:
            prompts: List of prompts to process
            system_message: Optional system message to use for all prompts
            
        Returns:
            List of responses
        """
        return [self.query(prompt, system_message) for prompt in prompts]
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if hasattr(self._query_with_cache, 'cache_clear'):
            self._query_with_cache.cache_clear()
            logger.info("Cache cleared")
            
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": self.cache_size,
            "cache_info": str(self._query_with_cache.cache_info()) if hasattr(self._query_with_cache, 'cache_info') else "No cache"
        }

    @classmethod
    def create(cls: type[T], **kwargs) -> T:
        """
        Factory method to create an instance of this client.
        
        Args:
            **kwargs: Arguments to pass to the constructor
            
        Returns:
            An instance of the client
        """
        return cls(**kwargs)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI models."""
    
    def __init__(self, 
                 model: Union[str, LLMModel] = LLMModel.GPT_3_5_TURBO, 
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 api_key: Optional[str] = None,
                 cache_size: int = 32):
        """
        Initialize the OpenAI client.
        
        Args:
            model: Model name or enum
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            api_key: OpenAI API key (if None, will be loaded from environment or file)
            cache_size: Size of response cache
        """
        self.api_key = api_key
        super().__init__(model, temperature, max_tokens, cache_size)
    
    def _setup(self) -> None:
        """Set up the OpenAI client."""
        # Try to load API key if not provided
        if not self.api_key:
            # First try environment variable
            self.api_key = os.getenv("OPENAI_API_KEY")
            
            # If not in environment, try key file
            if not self.api_key:
                key_files = [
                    Path(__file__).parent / "ChatGPT_key.txt",
                    Path(__file__).parent.parent / "prompt_processing" / "ChatGPT_key.txt"
                ]
                
                for key_file in key_files:
                    if key_file.exists():
                        try:
                            with open(key_file, 'r') as f:
                                self.api_key = f.read().strip()
                            logger.info(f"Loaded API key from {key_file}")
                            break
                        except Exception as e:
                            logger.warning(f"Error reading key file {key_file}: {str(e)}")
                
                if not self.api_key:
                    raise ConfigurationError("Failed to load OpenAI API key")
        
        # Set environment variable for LangChain
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Import LangChain modules here to avoid unnecessary imports if not using OpenAI
        try:
            from langchain.schema import HumanMessage, SystemMessage
            from langchain.chat_models import ChatOpenAI
            
            self.ChatOpenAI = ChatOpenAI
            self.HumanMessage = HumanMessage
            self.SystemMessage = SystemMessage
        except ImportError as e:
            raise ConfigurationError(f"Failed to import LangChain modules: {str(e)}")
    
    def _query_uncached(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Query the OpenAI model.
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            The model's response
        """
        # Create the chat model
        chat = self.ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            verbose=False  # Disable verbose logging
        )
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append(self.SystemMessage(content=system_message))
        messages.append(self.HumanMessage(content=prompt))
        
        try:
            response = chat.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise APIError(f"OpenAI API error: {str(e)}")


class OllamaClient(BaseLLMClient):
    """Client for Ollama models."""
    
    def __init__(self, 
                 model: Union[str, LLMModel] = LLMModel.LLAMA3, 
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 base_url: Optional[str] = None,
                 cache_size: int = 32,
                 streaming: bool = False):
        """
        Initialize the Ollama client.
        
        Args:
            model: Model name or enum
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            base_url: Base URL for Ollama API
            cache_size: Size of response cache
            streaming: Whether to stream responses
        """
        self.base_url = base_url or "http://localhost:11434"
        self.streaming = streaming
        super().__init__(model, temperature, max_tokens, cache_size)
    
    def _setup(self) -> None:
        """Set up the Ollama client."""
        # Import modules
        try:
            import requests
            self.requests = requests
            
            from langchain.callbacks.manager import CallbackManager
            from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
            from langchain_community.llms import Ollama
            
            self.CallbackManager = CallbackManager
            self.StreamingStdOutCallbackHandler = StreamingStdOutCallbackHandler
            self.Ollama = Ollama
        except ImportError as e:
            raise ConfigurationError(f"Failed to import required modules: {str(e)}")
        
        # Ensure Ollama is running
        if not self._ensure_ollama_running():
            raise ConfigurationError("Ollama server is not running and could not be started")
    
    def _ensure_ollama_running(self) -> bool:
        """
        Ensure Ollama is running and start it if not.
        
        Returns:
            True if Ollama is running, False otherwise
        """
        import subprocess
        import platform
        
        # Check if Ollama is running
        try:
            response = self.requests.get(f"{self.base_url}/api/version", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is already running")
                return True
        except Exception:
            logger.info("Ollama is not running. Attempting to start it...")
            
            # Ollama is not running, try to start it
            try:
                if platform.system() == 'Windows':
                    # Windows: Start in a new window
                    subprocess.Popen(['start', 'cmd', '/c', 'ollama serve'], shell=True)
                else:
                    # Unix-like systems (Linux, macOS): Start in background
                    subprocess.Popen(['ollama', 'serve'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
                
                # Wait for Ollama to start
                max_retries = 5
                for i in range(max_retries):
                    logger.info(f"Waiting for Ollama to start (attempt {i+1}/{max_retries})...")
                    time.sleep(2)
                    try:
                        response = self.requests.get(f"{self.base_url}/api/version", timeout=5)
                        if response.status_code == 200:
                            logger.info("Successfully started Ollama")
                            return True
                    except Exception:
                        continue
                
                logger.error("Failed to start Ollama after multiple attempts")
                return False
            except Exception as e:
                logger.error(f"Error starting Ollama: {str(e)}")
                return False
        
        return False
    
    def _query_uncached(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Query the Ollama model.
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            The model's response
        """
        # Create callback manager for streaming if enabled
        callback_manager = None
        if self.streaming:
            callback_manager = self.CallbackManager([self.StreamingStdOutCallbackHandler()])
        
        # Create the Ollama client
        ollama = self.Ollama(
            model=self.model_name,
            temperature=self.temperature,
            callback_manager=callback_manager,
            base_url=self.base_url
        )
        
        try:
            # Prepare combined prompt if system message is provided
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt}"
            else:
                full_prompt = prompt
                
            # Query the model
            response = ollama.invoke(full_prompt)
            return response
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise APIError(f"Ollama API error: {str(e)}")


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models."""
    
    def __init__(self, 
                 model: Union[str, LLMModel] = "claude-3-opus", 
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 api_key: Optional[str] = None,
                 cache_size: int = 32):
        """
        Initialize the Anthropic client.
        
        Args:
            model: Model name or enum
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            api_key: Anthropic API key
            cache_size: Size of response cache
        """
        self.api_key = api_key
        super().__init__(model, temperature, max_tokens, cache_size)
    
    def _setup(self) -> None:
        """Set up the Anthropic client."""
        # Try to load API key if not provided
        if not self.api_key:
            # Check environment variable
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            
            if not self.api_key:
                raise ConfigurationError("No Anthropic API key provided")
        
        # Ensure anthropic package is installed
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ConfigurationError("Anthropic package not installed. Install with 'pip install anthropic'")
    
    def _query_uncached(self, prompt: str, system_message: Optional[str] = None) -> str:
        """
        Query the Anthropic model.
        
        Args:
            prompt: The prompt to send
            system_message: Optional system message
            
        Returns:
            The model's response
        """
        client = self.anthropic.Anthropic(api_key=self.api_key)
        
        try:
            # Create message parameters
            params = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add system message if provided
            if system_message:
                params["system"] = system_message
                
            # Send request
            response = client.messages.create(**params)
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise APIError(f"Anthropic API error: {str(e)}")


class LLMClientFactory:
    """Factory for creating LLM clients."""
    
    _provider_mappings = {
        "openai": OpenAIClient,
        "gpt": OpenAIClient,
        "ollama": OllamaClient,
        "llama": OllamaClient,
        "anthropic": AnthropicClient,
        "claude": AnthropicClient
    }
    
    @classmethod
    def register_provider(cls, provider_name: str, client_class: type) -> None:
        """
        Register a new provider with the factory.
        
        Args:
            provider_name: Name of the provider
            client_class: Client class to use for this provider
        """
        cls._provider_mappings[provider_name.lower()] = client_class
    
    @classmethod
    def create_client(cls, provider: str, **kwargs) -> BaseLLMClient:
        """
        Create an LLM client for the specified provider.
        
        Args:
            provider: The LLM provider ('openai', 'ollama', etc.)
            **kwargs: Additional arguments for the client
            
        Returns:
            An LLM client instance
            
        Raises:
            ConfigurationError: If provider is unsupported
        """
        provider = provider.lower()
        
        if provider in cls._provider_mappings:
            client_class = cls._provider_mappings[provider]
            return client_class(**kwargs)
        else:
            supported = ", ".join(cls._provider_mappings.keys())
            raise ConfigurationError(f"Unsupported LLM provider: {provider}. Supported providers: {supported}")


# Simple usage example
if __name__ == "__main__":
    # Create clients
    openai_client = LLMClientFactory.create_client("openai")
    ollama_client = LLMClientFactory.create_client("ollama")
    
    # Example prompt
    prompt = "Explain the concept of recursion in simple terms."
    
    # Query both models
    print("=== OpenAI Response ===")
    openai_response = openai_client.query(prompt)
    print(f"\n{openai_response}\n")
    
    print("=== Ollama Response ===")
    ollama_response = ollama_client.query(prompt)
    print(f"\n{ollama_response}\n")
    
    # Example of async usage
    async def async_example():
        print("=== Async Example ===")
        prompts = [
            "What is machine learning?",
            "Explain natural language processing.",
            "What are neural networks?"
        ]
        
        results = await openai_client.abatch_query(prompts)
        for i, result in enumerate(results):
            print(f"\nQuestion {i+1}: {prompts[i]}")
            print(f"Answer: {result[:100]}...\n")
    
    # Uncomment to run async example
    # import asyncio
    # asyncio.run(async_example()) 