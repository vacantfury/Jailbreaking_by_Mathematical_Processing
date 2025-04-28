"""
LLM service module for invoking different language models.
"""
import os
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, cast, TYPE_CHECKING
from pathlib import Path
import logging
from functools import lru_cache
import hashlib
import requests
import platform
import subprocess
import importlib.util
import weakref
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('llm_service')

# Global thread pool for async operations
_THREAD_POOL = ThreadPoolExecutor(max_workers=4)

# Import constants and model enum
from .constants import (
    LLM_CONFIG, 
    SYSTEM_MESSAGE, 
    DEFAULT_TEMPERATURE, 
    DEFAULT_MAX_TOKENS,
    OLLAMA_BASE_URL, 
    OLLAMA_VERSION_URL,
    PROMPT_TEMPLATE,
)
from .llm_model import LLMModel

# Lazy imports for LangChain
_langchain_loaded = False
_langchain_modules = {}

def _import_langchain():
    """Lazily import LangChain modules to improve startup time."""
    global _langchain_loaded, _langchain_modules
    
    if not _langchain_loaded:
        try:
            from langchain.llms import OpenAI
            from langchain.chat_models import ChatOpenAI
            from langchain.llms.base import LLM
            from langchain.schema import HumanMessage, SystemMessage
            from langchain.callbacks.manager import CallbackManager
            from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
            from langchain_community.llms import Ollama
            
            _langchain_modules = {
                'OpenAI': OpenAI,
                'ChatOpenAI': ChatOpenAI,
                'LLM': LLM, 
                'HumanMessage': HumanMessage,
                'SystemMessage': SystemMessage,
                'CallbackManager': CallbackManager,
                'StreamingStdOutCallbackHandler': StreamingStdOutCallbackHandler,
                'Ollama': Ollama
            }
            _langchain_loaded = True
            return True
        except ImportError as e:
            logger.error(f"Failed to import LangChain modules: {e}")
            return False
    return True


class ServiceError(Exception):
    """Base exception for LLM service errors."""
    pass


class ConfigurationError(ServiceError):
    """Exception for configuration errors."""
    pass


class APIError(ServiceError):
    """Exception for API errors."""
    pass


class LLMService:
    """Service class for invoking LLMs."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_size: int = 128, cache_responses: bool = True):
        """
        Initialize the LLM service.
        
        Args:
            cache_size: Maximum number of responses to cache
            cache_responses: Whether to cache responses
        """
        # Skip initialization if already done
        if getattr(self, "_initialized", False):
            return
            
        # Cache for LLM clients - use weak references to allow garbage collection
        self._clients = weakref.WeakValueDictionary()
        self._cache_responses = cache_responses
        self._cache_size = cache_size
        
        # Initialize the cache function with the right size
        # Remove the class decorator and create a fresh one with our cache size
        self._call_llm_cached = lru_cache(maxsize=self._cache_size)(self._call_llm_uncached)
        
        # Set the initialized flag
        self._initialized = True
        
        logger.info("LLM service initialized")
    
    def invoke_llm(self, 
                  prompt: str, 
                  model: Union[str, LLMModel],
                  system_message: Optional[str] = None,
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> str:
        """
        Invoke an LLM with the given prompt and parameters.
        
        Args:
            prompt: The user prompt to process
            model: The LLM model to use (string or enum)
            system_message: Optional system message
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            The LLM's response
        
        Raises:
            ConfigurationError: If there's a configuration issue
            APIError: If there's an error calling the API
            ServiceError: For other service errors
        """
        # Convert string model to enum if needed
        if isinstance(model, str):
            try:
                model = LLMModel.from_string(model)
            except ValueError:
                raise ConfigurationError(f"Unknown model: {model}")
                
        # Use defaults if parameters not provided
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        system_message = system_message or SYSTEM_MESSAGE
        
        # Get input hash for caching
        input_hash = self._get_input_hash(prompt, model, system_message, temperature, max_tokens)
        
        try:
            if self._cache_responses:
                return self._call_llm_cached(input_hash, prompt, model, system_message, temperature, max_tokens)
            else:
                return self._call_llm_uncached(input_hash, prompt, model, system_message, temperature, max_tokens)
        except Exception as e:
            if isinstance(e, (ConfigurationError, APIError)):
                raise
            logger.error(f"Error invoking LLM: {str(e)}")
            raise ServiceError(f"Failed to invoke LLM: {str(e)}")
    
    async def ainvoke_llm(self, 
                         prompt: str, 
                         model: Union[str, LLMModel],
                         system_message: Optional[str] = None,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> str:
        """
        Asynchronously invoke an LLM with the given prompt and parameters.
        
        Args:
            prompt: The user prompt to process
            model: The LLM model to use
            system_message: Optional system message
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            
        Returns:
            The LLM's response
        """
        # Convert string model to enum if needed
        if isinstance(model, str):
            try:
                model = LLMModel.from_string(model)
            except ValueError:
                raise ConfigurationError(f"Unknown model: {model}")
                
        # Use defaults if parameters not provided
        temperature = temperature if temperature is not None else DEFAULT_TEMPERATURE
        system_message = system_message or SYSTEM_MESSAGE
        
        # Get input hash for caching
        input_hash = self._get_input_hash(prompt, model, system_message, temperature, max_tokens)
        
        # Run the synchronous version in a thread pool
        loop = asyncio.get_event_loop()
        if self._cache_responses:
            return await loop.run_in_executor(
                _THREAD_POOL, 
                lambda: self._call_llm_cached(input_hash, prompt, model, system_message, temperature, max_tokens)
            )
        else:
            return await loop.run_in_executor(
                _THREAD_POOL,
                lambda: self._call_llm_uncached(input_hash, prompt, model, system_message, temperature, max_tokens)
            )
    
    def _get_input_hash(self, prompt, model, system_message, temperature, max_tokens):
        """Generate a hash for the input parameters."""
        model_value = model.value if isinstance(model, LLMModel) else str(model)
        hash_input = f"{prompt}|{model_value}|{system_message}|{temperature}|{max_tokens}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _call_llm_cached(self, 
                       input_hash: str,
                       prompt: str, 
                       model: LLMModel,
                       system_message: str,
                       temperature: float,
                       max_tokens: Optional[int]) -> str:
        """Cached version of LLM call."""
        return self._call_llm_uncached(input_hash, prompt, model, system_message, temperature, max_tokens)
    
    def _call_llm_uncached(self, 
                          input_hash: str,
                          prompt: str, 
                          model: LLMModel,
                          system_message: str,
                          temperature: float,
                          max_tokens: Optional[int]) -> str:
        """
        Call the LLM without caching.
        
        Args:
            input_hash: Hash of input for caching
            prompt: The user prompt
            model: The LLM model
            system_message: System message
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            LLM response
        """
        # Ensure LangChain is imported when needed
        if not _import_langchain():
            raise ConfigurationError("LangChain is required but not available")
        
        try:
            # Get or create the client
            client = self._get_client(model, temperature, max_tokens)
            
            # Invoke the appropriate client based on provider
            provider = model.provider
            if provider == "openai":
                return self._invoke_openai(client, prompt, system_message)
            elif provider == "ollama":
                return self._invoke_ollama(client, prompt, system_message)
            else:
                raise ConfigurationError(f"Unsupported provider: {provider}")
        except Exception as e:
            if isinstance(e, (ConfigurationError, APIError)):
                raise
            logger.error(f"Error in LLM call: {str(e)}")
            raise ServiceError(f"LLM call failed: {str(e)}")
    
    def _get_client(self, 
                   model: LLMModel, 
                   temperature: float, 
                   max_tokens: Optional[int]) -> Any:
        """
        Get a cached client or create a new one.
        
        Args:
            model: The LLM model
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            LLM client
        """
        # Create a cache key
        cache_key = f"{model.value}_{temperature}_{max_tokens}"
        
        # Return cached client if exists
        if cache_key in self._clients:
            return self._clients[cache_key]
        
        # Create new client based on provider
        if model.provider == "openai":
            client = self._create_openai_client(model.value, temperature, max_tokens)
        elif model.provider == "ollama":
            client = self._create_ollama_client(model.value, temperature)
        else:
            raise ConfigurationError(f"Unsupported provider for model {model}")
        
        # Cache the client
        self._clients[cache_key] = client
        return client
    
    def _create_openai_client(self, 
                            model_name: str, 
                            temperature: float,
                            max_tokens: Optional[int]) -> Any:
        """
        Create an OpenAI client.
        
        Args:
            model_name: Name of the model
            temperature: Temperature setting
            max_tokens: Maximum tokens
            
        Returns:
            ChatOpenAI instance
        """
        # Load API key
        api_key = os.getenv("OPENAI_API_KEY")
        
        # If not in environment, try to find in project
        if not api_key:
            try:
                key_file = Path(__file__).parent / "ChatGPT_key.txt"
                if key_file.exists():
                    with open(key_file, 'r') as f:
                        api_key = f.read().strip()
                    os.environ["OPENAI_API_KEY"] = api_key
                    logger.info("Loaded API key from file")
                else:
                    raise ConfigurationError("API key file not found")
            except Exception as e:
                raise ConfigurationError(f"Failed to load OpenAI API key: {str(e)}")
        
        # Create client
        try:
            ChatOpenAI = _langchain_modules.get('ChatOpenAI')
            return ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=False  # Disable verbose logging
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create OpenAI client: {str(e)}")
    
    def _create_ollama_client(self, 
                            model_name: str, 
                            temperature: float) -> Any:
        """
        Create an Ollama client.
        
        Args:
            model_name: Name of the model
            temperature: Temperature setting
            
        Returns:
            Ollama instance
        """
        # Ensure Ollama is running
        if not self._ensure_ollama_running():
            raise ConfigurationError("Ollama server is not running and could not be started")
        
        # Create callback manager for streaming
        try:
            CallbackManager = _langchain_modules.get('CallbackManager')
            StreamingStdOutCallbackHandler = _langchain_modules.get('StreamingStdOutCallbackHandler')
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            
            # Create client
            Ollama = _langchain_modules.get('Ollama')
            return Ollama(
                model=model_name,
                temperature=temperature,
                callback_manager=callback_manager,
                base_url=OLLAMA_BASE_URL
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to create Ollama client: {str(e)}")
    
    def _invoke_openai(self, 
                      client: Any, 
                      prompt: str, 
                      system_message: str) -> str:
        """
        Invoke OpenAI model.
        
        Args:
            client: OpenAI client
            prompt: User prompt
            system_message: System message
            
        Returns:
            Model response
        """
        try:
            SystemMessage = _langchain_modules.get('SystemMessage')
            HumanMessage = _langchain_modules.get('HumanMessage')
            
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt)
            ]
            
            response = client.invoke(messages)
            return response.content
        except Exception as e:
            logger.error(f"Error invoking OpenAI: {str(e)}")
            raise APIError(f"Error invoking OpenAI: {str(e)}")
    
    def _invoke_ollama(self, 
                      client: Any, 
                      prompt: str, 
                      system_message: str) -> str:
        """
        Invoke Ollama model.
        
        Args:
            client: Ollama client
            prompt: User prompt
            system_message: System message
            
        Returns:
            Model response
        """
        try:
            # Format prompt with system message
            formatted_prompt = f"{system_message}\n\n{prompt}"
            
            # Invoke model
            response = client.invoke(formatted_prompt)
            return response
        except Exception as e:
            logger.error(f"Error invoking Ollama: {str(e)}")
            raise APIError(f"Error invoking Ollama: {str(e)}")
    
    def _ensure_ollama_running(self) -> bool:
        """
        Ensure Ollama is running and start it if not.
        
        Returns:
            True if Ollama is running, False otherwise
        """
        # Check if Ollama is running
        try:
            response = requests.get(OLLAMA_VERSION_URL, timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is already running")
                return True
        except requests.exceptions.RequestException:
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
                        response = requests.get(OLLAMA_VERSION_URL, timeout=5)
                        if response.status_code == 200:
                            logger.info("Successfully started Ollama")
                            return True
                    except requests.exceptions.RequestException:
                        continue
                
                logger.error("Failed to start Ollama after multiple attempts")
                return False
            except Exception as e:
                logger.error(f"Error starting Ollama: {str(e)}")
                return False
        
        return False
    
    def format_prompt(self, 
                     system_message: str, 
                     user_prompt: str, 
                     previous_example: Optional[str] = None) -> str:
        """
        Format a prompt for sending to an LLM.
        
        Args:
            system_message: System message (instructions)
            user_prompt: User's prompt
            previous_example: Optional example for few-shot learning
            
        Returns:
            Formatted prompt string
        """
        if previous_example:
            return PROMPT_TEMPLATE.format(
                system_message=system_message,
                previous_example=previous_example,
                user_prompt=user_prompt
            )
        else:
            # Simpler template without example
            return f"{system_message}\n\nUser: {user_prompt}"
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        if hasattr(self._call_llm_cached, 'cache_clear'):
            # Get cache info before clearing for logging
            if hasattr(self._call_llm_cached, 'cache_info'):
                cache_info = self._call_llm_cached.cache_info()
                logger.info(f"Clearing response cache with {cache_info.currsize} entries")
            else:
                logger.info("Clearing response cache")
                
            # Clear the cache
            self._call_llm_cached.cache_clear()
            logger.info("Response cache cleared")
        else:
            logger.warning("No cache to clear")
    
    def client_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the client cache.
        
        Returns:
            Dictionary with client cache statistics
        """
        cache_info = {}
        if hasattr(self._call_llm_cached, 'cache_info'):
            info = self._call_llm_cached.cache_info()
            cache_info = {
                "hits": info.hits,
                "misses": info.misses,
                "maxsize": info.maxsize,
                "currsize": info.currsize
            }
            
        return {
            "client_count": len(self._clients),
            "cache_size": self._cache_size if self._cache_responses else 0,
            "cache_enabled": self._cache_responses,
            "cache_info": cache_info
        }


# Create global instance
llm_service = LLMService() 