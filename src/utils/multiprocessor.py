"""
Multiprocessing utilities for parallel execution.

This module provides utilities for executing functions in parallel across multiple processes,
optimized for M4 chip with 48GB RAM.
"""
import multiprocessing as mp
from typing import Callable, List, Any, Optional, Tuple
from functools import partial
import traceback

from src.utils.constants import (
    DEFAULT_NUM_WORKERS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_WORKER_TIMEOUT,
    MULTIPROCESS_START_METHOD,
    SHOW_MULTIPROCESS_PROGRESS,
)


def _worker_wrapper(func: Callable, item: Any, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """
    Wrapper function to catch exceptions in worker processes.
    
    Args:
        func: Function to execute
        item: Item from the collection to process
        *args: Additional positional arguments for func
        **kwargs: Additional keyword arguments for func
    
    Returns:
        Tuple of (result, exception). If successful, exception is None.
    """
    try:
        result = func(item, *args, **kwargs)
        return (result, None)
    except Exception as e:
        # Capture the full traceback for debugging
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        return (None, error_msg)


def multiprocess_run(
    func: Callable,
    main_list: List[Any],
    task_type: str = "cpu",
    chunk_size: Optional[int] = None,
    timeout: Optional[int] = DEFAULT_WORKER_TIMEOUT,
    show_progress: bool = SHOW_MULTIPROCESS_PROGRESS,
    handle_errors: str = "raise",
    sequential: bool = False,
    **kwargs
) -> List[Any]:
    """
    Execute a function on a list of items in parallel using multiprocessing.
    
    Args:
        func: Function to execute. Should take an item from main_list as first argument.
        main_list: List of items to process in parallel.
        task_type: Type of task - "cpu" (default) or "io"
        chunk_size: Size of chunks for imap. Defaults to None (auto-calculated).
        timeout: Timeout in seconds. Defaults to DEFAULT_WORKER_TIMEOUT.
        show_progress: Whether to show progress. Defaults to SHOW_MULTIPROCESS_PROGRESS.
        handle_errors: How to handle errors: "raise" (default), "skip", or "collect".
        sequential: Force sequential execution. Defaults to False.
        **kwargs: Additional arguments, e.g., num_workers override.
    """
    # Extract extra config from kwargs if any
    manual_workers = kwargs.get("num_workers", None)

    if not main_list:
        return []

    # Check if we are already in a daemon process (nested pool is not allowed)
    # This prevents "AssertionError: daemonic processes are not allowed to have children"
    if mp.current_process().daemon and not sequential:
        # if show_progress:
        #     print(f"Running in daemon process (worker), forcing sequential execution for {len(main_list)} items...")
        sequential = True

    # Check for sequential execution request
    if sequential:
        if show_progress:
            print(f"Processing {len(main_list)} items sequentially (forced)...")
        results = []
        for i, item in enumerate(main_list):
            if show_progress and len(main_list) > 1 and (i + 1) % max(1, len(main_list) // 10) == 0:
                print(f"  Progress: {i+1}/{len(main_list)}")
            
            result, error = _worker_wrapper(func, item)
            
            if error:
                if handle_errors == "raise":
                    raise RuntimeError(f"Error processing item {i}: {error}")
                elif handle_errors == "collect":
                    results.append((result, error))
            else:
                if handle_errors == "collect":
                    results.append((result, None))
                else:
                    results.append(result)
        return results
    
    # Automatically determine optimal number of workers
    num_workers = get_optimal_worker_count(len(main_list), task_type=task_type, max_workers=manual_workers)
    
    # Use single process if only one item or one worker
    if len(main_list) == 1 or num_workers == 1:
        if show_progress:
            print(f"Processing {len(main_list)} items sequentially...")
        results = []
        for i, item in enumerate(main_list):
            if show_progress and len(main_list) > 1:
                print(f"  Progress: {i+1}/{len(main_list)}")
            result, error = _worker_wrapper(func, item)
            if error and handle_errors == "raise":
                raise RuntimeError(f"Error processing item {i}: {error}")
            if handle_errors == "collect":
                results.append((result, error))
            elif error is None:
                results.append(result)
        return results
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = max(1, len(main_list) // (num_workers * 4))
        chunk_size = min(chunk_size, DEFAULT_CHUNK_SIZE)
    
    if timeout is None:
        timeout = DEFAULT_WORKER_TIMEOUT
    
    if show_progress:
        print(f"Processing {len(main_list)} items with {num_workers} workers (chunk_size={chunk_size})...")
    
    # Create partial function with wrapper
    worker_func = partial(_worker_wrapper, func)
    
    # Use spawn method for better compatibility with CUDA/torch
    ctx = mp.get_context(MULTIPROCESS_START_METHOD)
    
    results = []
    errors_occurred = False
    pool = None
    
    try:
        pool = ctx.Pool(processes=num_workers)
        # Use imap for better memory efficiency with large lists
        for i, (result, error) in enumerate(pool.imap(worker_func, main_list, chunksize=chunk_size)):
            if show_progress and (i + 1) % max(1, len(main_list) // 10) == 0:
                print(f"  Progress: {i+1}/{len(main_list)}")
            
            if error:
                errors_occurred = True
                if handle_errors == "raise":
                    pool.terminate()
                    raise RuntimeError(f"Error processing item {i}: {error}")
                elif handle_errors == "collect":
                    results.append((result, error))
                # If "skip", do nothing
            else:
                if handle_errors == "collect":
                    results.append((result, None))
                else:
                    results.append(result)
        
        pool.close()
        pool.join()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating workers...")
        if pool:
            pool.terminate()
            pool.join()
        raise
    except Exception as e:
        print(f"\nError in multiprocessing: {e}")
        if pool:
            try:
                pool.terminate()
                pool.join()
            except Exception:
                pass
        raise
    
    if show_progress:
        if errors_occurred and handle_errors != "raise":
            print(f"Completed with some errors. Processed {len(results)}/{len(main_list)} items.")
        else:
            print(f"Completed processing {len(results)} items.")
    
    return results


def get_optimal_worker_count(
    num_items: int,
    task_type: str = "cpu",
    max_workers: Optional[int] = None
) -> int:
    """
    Calculate optimal number of workers based on task type and item count.
    
    Args:
        num_items: Number of items to process.
        task_type: Type of task - "cpu" for CPU-bound, "io" for I/O-bound.
        max_workers: Maximum number of workers to use.
    
    Returns:
        Optimal number of workers.
    """
    if task_type == "cpu":
        base_workers = DEFAULT_NUM_WORKERS
    elif task_type == "io":
        base_workers = DEFAULT_NUM_WORKERS * 2
    else:
        raise ValueError(f"Invalid task_type: {task_type}. Use 'cpu' or 'io'.")
    
    # Don't use more workers than items
    optimal = min(base_workers, num_items)
    
    # Apply max_workers limit if specified
    if max_workers is not None:
        optimal = min(optimal, max_workers)
    
    return max(1, optimal)

