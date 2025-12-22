"""
LLM model definitions with provider information.
"""
from enum import Enum


class Provider(str, Enum):
    """Enum for LLM service providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    TRANSFORMERS = "transformers"


class LLMModel(Enum):
    """
    Enum for LLM models.
    
    Each model has:
    - model_id: The actual model identifier string
    - provider: Which service implementation to use
    """
    
    # OpenAI models
    GPT_3_5_TURBO = ("gpt-3.5-turbo", Provider.OPENAI)
    GPT_4 = ("gpt-4", Provider.OPENAI)
    GPT_4_TURBO = ("gpt-4-turbo", Provider.OPENAI)
    GPT_4O = ("gpt-4o", Provider.OPENAI)
    GPT_4_1 = ("gpt-4.1", Provider.OPENAI)
    GPT_4_1_MINI = ("gpt-4.1-mini", Provider.OPENAI)
    GPT_4_1_NANO = ("gpt-4.1-nano", Provider.OPENAI)
    GPT_4_NANO = ("gpt-4-nano", Provider.OPENAI)
    GPT_O3 = ("gpt-o3", Provider.OPENAI)
    GPT_O4_MINI = ("gpt-o4-mini", Provider.OPENAI)
    GPT_5 = ("gpt-5", Provider.OPENAI)
    GPT_5_MINI = ("gpt-5-mini", Provider.OPENAI)
    GPT_5_NANO = ("gpt-5-nano", Provider.OPENAI)
    GEMINI_FLASH = ("gemini-2.0-flash-exp", Provider.GOOGLE)
    GEMINI_PRO = ("gemini-2.0-pro-exp", Provider.GOOGLE)  # Verified available
    
    # Validated with verify_anthropic_models.py
    # Note: 3.5 Sonnet is NOT available for this user.
    CLAUDE_3_5_SONNET = ("claude-3-5-sonnet-20241022", Provider.ANTHROPIC) # Deprecated/Unavailable
    CLAUDE_3_5_HAIKU = ("claude-3-5-haiku-20241022", Provider.ANTHROPIC)
    CLAUDE_3_7_SONNET = ("claude-3-7-sonnet-20250219", Provider.ANTHROPIC) # Verified available
    
    # Placeholder for generic GPT-4 Nano if needed
    CLAUDE_4_1_OPUS = ("claude-3-opus-20240229", Provider.ANTHROPIC) # Replaced latest
    CLAUDE_4_5_SONNET = ("claude-3-5-sonnet-20241022", Provider.ANTHROPIC) # Replaced latest
    # cheapest among the newest claude models
    CLAUDE_4_5_HAIKU = ("claude-3-5-haiku-20241022", Provider.ANTHROPIC) # Replaced latest
    
    # Specific versions pinned to avoid 'latest' aliases
    GEMINI_1_5_FLASH = ("gemini-1.5-flash-002", Provider.GOOGLE)
    GEMINI_1_5_FLASH_8B = ("gemini-1.5-flash-8b-001", Provider.GOOGLE) 
    GEMINI_1_5_PRO = ("gemini-1.5-pro-002", Provider.GOOGLE)
    GEMINI_2_0_FLASH = ("gemini-2.0-flash", Provider.GOOGLE)
    GEMINI_2_0_FLASH_LITE = ("gemini-2.0-flash-lite", Provider.GOOGLE)
    GEMINI_2_5_FLASH = ("gemini-2.5-flash", Provider.GOOGLE)
    GEMINI_2_5_FLASH_LITE = ("gemini-2.5-flash-lite", Provider.GOOGLE)
    GEMINI_2_5_FLASH_PRO = ("gemini-2.5-pro", Provider.GOOGLE)
    GEMINI_3_PRO_PREVIEW = ("gemini-3-pro-preview", Provider.GOOGLE)
    
    LLAMA3 = ("llama3", Provider.TRANSFORMERS)
    LLAMA3_1 = ("llama3.1", Provider.TRANSFORMERS)
    MISTRAL = ("mistral", Provider.TRANSFORMERS)
    
    # Transformers models (local HuggingFace)
    # Llama 3 series (8B)
    LLAMA3_8B = ("meta-llama/Meta-Llama-3-8B-Instruct", Provider.TRANSFORMERS)
    LLAMA3_2_1B = ("meta-llama/Llama-3.2-1B-Instruct", Provider.TRANSFORMERS)  # ~2GB, very fast
    LLAMA3_2_3B = ("meta-llama/Llama-3.2-3B-Instruct", Provider.TRANSFORMERS)  # ~6GB, balanced
    
    # Other models
    MISTRAL_7B = ("mistralai/Mistral-7B-Instruct-v0.2", Provider.TRANSFORMERS)
    PHI_3_MINI = ("microsoft/Phi-3-mini-4k-instruct", Provider.TRANSFORMERS)  # ~7GB, good quality
    
    @property
    def model_id(self) -> str:
        """Get the model identifier string."""
        return self.value[0]
    
    @property
    def provider(self) -> Provider:
        """Get the provider for this model."""
        return self.value[1]
    
