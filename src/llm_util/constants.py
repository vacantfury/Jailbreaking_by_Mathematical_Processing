"""
Constants for LLM configurations and endpoints.
"""
from typing import Dict, Any, Final

# API endpoints
OPENAI_API_URL: Final[str] = "https://api.openai.com/v1"
OLLAMA_BASE_URL: Final[str] = "http://localhost:11434"
OLLAMA_VERSION_URL: Final[str] = f"{OLLAMA_BASE_URL}/api/version"
OLLAMA_CHAT_URL: Final[str] = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_GENERATE_URL: Final[str] = f"{OLLAMA_BASE_URL}/api/generate"

# Default parameters for LLM requests
DEFAULT_MAX_TOKENS: Final[int] = 1000
DEFAULT_TEMPERATURE: Final[float] = 0.7

# LLM configuration
LLM_CONFIG: Final[Dict[str, Dict[str, Any]]] = {
    "gpt": {
        "model": "gpt-3.5-turbo",
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "api_endpoint": OPENAI_API_URL
    },
    "llama": {
        "model": "llama3",
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "api_endpoint": OLLAMA_CHAT_URL
    }
}

# Default system message for prompts
SYSTEM_MESSAGE: Final[str] = "You are a helpful assistant."

# Template for few-shot learning prompts
PROMPT_TEMPLATE: Final[str] = """
{system_message}

{previous_example}

User: {user_prompt}
""" 