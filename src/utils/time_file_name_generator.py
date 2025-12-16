"""
Utility for generating unique filenames with timestamps and random identifiers.
"""
import random
from datetime import datetime
from typing import Optional


def generate_timestamped_filename(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    timestamp_format: str = "%Y%m%d_%H%M%S"
) -> str:
    """
    Generate a unique filename string with timestamp and random identifier.
    
    Format: [prefix_]<timestamp>_<random_int>[_suffix]
    Example: "exp_20241013_143025_45678901.jsonl"
    
    Args:
        prefix: Optional prefix before timestamp (e.g., "exp", "task")
        suffix: Optional suffix after random int (e.g., ".jsonl", "_results")
        timestamp_format: Format for timestamp (default: YYYYMMDD_HHMMSS)
    
    Returns:
        Unique filename string
        
    Example:
        >>> generate_timestamped_filename()
        '20241013_143025_45678901'
        
        >>> generate_timestamped_filename(prefix="exp", suffix=".jsonl")
        'exp_20241013_143025_45678901.jsonl'
        
        >>> generate_timestamped_filename(prefix="task", suffix="_results")
        'task_20241013_143025_45678901_results'
    """
    # Generate timestamp
    timestamp = datetime.now().strftime(timestamp_format)
    
    # Generate random int between 0 and 100000000
    random_int = random.randint(0, 100000000)
    
    # Build filename
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(timestamp)
    parts.append(str(random_int))
    if suffix:
        parts.append(suffix)
    
    # Join with underscore, but handle suffix specially if it starts with a dot
    if suffix and suffix.startswith('.'):
        # For file extensions, don't add underscore before them
        main_parts = parts[:-1]
        return '_'.join(main_parts) + suffix
    else:
        return '_'.join(parts)


def generate_experiment_filename(experiment_name: Optional[str] = None) -> str:
    """
    Generate a filename specifically for experiment results.
    
    Args:
        experiment_name: Optional experiment name to include
    
    Returns:
        Filename string with .jsonl extension
        
    Example:
        >>> generate_experiment_filename()
        'exp_20241013_143025_45678901.jsonl'
        
        >>> generate_experiment_filename("jailbreak_test")
        'exp_jailbreak_test_20241013_143025_45678901.jsonl'
    """
    if experiment_name:
        prefix = f"exp_{experiment_name}"
    else:
        prefix = "exp"
    
    return generate_timestamped_filename(prefix=prefix, suffix=".jsonl")


def generate_task_filename(task_name: Optional[str] = None) -> str:
    """
    Generate a filename specifically for task results.
    
    Format: {timestamp}_{random}_{task_name}.jsonl
    
    Args:
        task_name: Optional task name to include
    
    Returns:
        Filename string with .jsonl extension
        
    Example:
        >>> generate_task_filename()
        '20241013_143025_45678901.jsonl'
        
        >>> generate_task_filename("split_reassemble")
        '20241013_143025_45678901_split_reassemble.jsonl'
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate random int between 0 and 100000000
    random_int = random.randint(0, 100000000)
    
    # Build filename: {timestamp}_{random}_{task_name}.jsonl
    if task_name:
        filename = f"{timestamp}_{random_int}_{task_name}.jsonl"
    else:
        filename = f"{timestamp}_{random_int}.jsonl"
    
    return filename


