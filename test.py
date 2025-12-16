import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

# Import src package (handles platform-specific fixes like OpenMP)
import src  # noqa: F401

from src.llm_utils import LLMServiceFactory, LLMModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


def test_openai_service():
    """Test the OpenAI service functionality."""
    print("\n" + "="*60)
    logger.info("Testing OpenAI Service...")
    print("="*60)
    
    try:
        # Create OpenAI service
        service = LLMServiceFactory.create(
            LLMModel.GPT_4,
            temperature=0.0,
            max_tokens=100
        )
        
        logger.info(f"Created service: {service}")
        
        # Test batch_generate
        logger.info("Testing batch_generate...")
        test_prompts = [
            ("test_001", "What is 2+2?"),
            ("test_002", "What is the capital of France?"),
        ]
        
        results = service.batch_generate(
            prompts=test_prompts,
            system_message="You are a helpful assistant."
        )
        
        logger.info(f"Generated {len(results)} responses")
        for prompt_id, response in results:
            logger.info(f"  [{prompt_id}]: {response[:100]}...")
        
        # Test batch_chat
        logger.info("Testing batch_chat...")
        test_conversations = [
            ("conv_001", [
                ("Hello, who are you?", None),
            ]),
        ]
        
        chat_results = service.batch_chat(conversations=test_conversations)
        
        logger.info(f"Generated {len(chat_results)} chat responses")
        for conv_id, response in chat_results:
            logger.info(f"  [{conv_id}]: {response[:100]}...")
        
        logger.info("OpenAI service test completed successfully!")
        return True
        
    except ValueError as e:
        logger.warning(f"API key not set: {e}")
        logger.warning("Set OPENAI_API_KEY in .env file to run this test")
        return False
    except Exception as e:
        logger.error(f"OpenAI service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_claude_service():
    """Test the Claude (Anthropic) service functionality."""
    print("\n" + "="*60)
    logger.info("Testing Claude Service...")
    print("="*60)
    
    try:
        # Create Claude service
        service = LLMServiceFactory.create(
            LLMModel.CLAUDE_3_5_SONNET,
            temperature=0.0,
            max_tokens=100
        )
        
        logger.info(f"Created service: {service}")
        
        # Test batch_generate
        logger.info("Testing batch_generate...")
        test_prompts = [
            ("test_001", "What is 2+2?"),
            ("test_002", "What is the capital of Japan?"),
        ]
        
        results = service.batch_generate(
            prompts=test_prompts,
            system_message="You are a helpful assistant."
        )
        
        logger.info(f"Generated {len(results)} responses")
        for prompt_id, response in results:
            logger.info(f"  [{prompt_id}]: {response[:100]}...")
        
        # Test batch_chat
        logger.info("Testing batch_chat...")
        test_conversations = [
            ("conv_001", [
                ("Hello, who are you?", None),
            ]),
        ]
        
        chat_results = service.batch_chat(conversations=test_conversations)
        
        logger.info(f"Generated {len(chat_results)} chat responses")
        for conv_id, response in chat_results:
            logger.info(f"  [{conv_id}]: {response[:100]}...")
        
        logger.info("Claude service test completed successfully!")
        return True
        
    except ValueError as e:
        logger.warning(f"API key not set: {e}")
        logger.warning("Set ANTHROPIC_API_KEY in .env file to run this test")
        return False
    except Exception as e:
        logger.error(f"Claude service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_local_llama_service():
    """Test the Local Llama service functionality."""
    print("\n" + "="*60)
    logger.info("Testing Local Llama Service...")
    print("="*60)
    
    try:
        # Create Local Llama service
        service = LLMServiceFactory.create(
            LLMModel.LLAMA3_8B,
            temperature=0.0,
            max_tokens=100
        )
        
        logger.info(f"Created service: {service}")
        
        # Test batch_generate with 30 identical prompts
        logger.info("Testing batch_generate with 30 identical prompts...")
        test_prompts = [
            (f"test_{i:03d}", "What is 2+2? Answer briefly.")
            for i in range(1, 31)
        ]
        
        logger.info(f"Processing {len(test_prompts)} prompts...")
        results = service.batch_generate(
            prompts=test_prompts,
            system_message="You are a helpful assistant. Answer concisely."
        )
        
        logger.info(f"Generated {len(results)} responses")
        # Show first 5 and last 5 responses
        for i, (prompt_id, response) in enumerate(results):
            if i < 5 or i >= len(results) - 5:
                logger.info(f"  [{prompt_id}]: {response[:200]}...")
            elif i == 5:
                logger.info(f"  ... ({len(results) - 10} more responses) ...")
        
        logger.info("Local Llama service test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Local Llama service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test(mode: str):
    """Run a test based on the specified mode.
    
    Args:
        mode: Type of test to run - "openai", "claude", "llama", or "all"
    """
    
    results = {}
    
    if mode.lower() == "openai":
        results['openai'] = test_openai_service()
    elif mode.lower() == "claude":
        results['claude'] = test_claude_service()
    elif mode.lower() == "llama":
        results['llama'] = test_local_llama_service()
    elif mode.lower() == "all":
        results['openai'] = test_openai_service()
        results['claude'] = test_claude_service()
        results['llama'] = test_local_llama_service()
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'openai', 'claude', 'llama', or 'all'")
    
    # Print summary
    print("\n" + "="*60)
    logger.info("TEST SUMMARY")
    print("="*60)
    for service, success in results.items():
        status = "PASSED" if success else "FAILED"
        if success:
            logger.info(f"{service.upper()}: {status}")
        else:
            logger.error(f"{service.upper()}: {status}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    test("llama")

