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


def parallel_map(
    func: Callable,
    items: List[Any],
    task_type: str = "cpu",
    chunk_size: Optional[int] = None,
    timeout: Optional[int] = None,
    show_progress: bool = SHOW_MULTIPROCESS_PROGRESS,
    handle_errors: str = "raise",
    sequential: bool = False,
) -> List[Any]:
    """
    Execute a function on a list of items in parallel using multiprocessing.
    
    This function distributes the items across multiple worker processes and collects
    the results in order. It's optimized for M4 chip with 48GB RAM.
    Worker count is automatically determined based on task type and number of items.
    
    Args:
        func: Function to execute. Should take an item from items as first argument.
              Use functools.partial to pass additional arguments.
        items: List of items to process in parallel.
        task_type: Type of task - "cpu" for CPU-bound (default), "io" for I/O-bound.
        chunk_size: Size of chunks for imap. Defaults to auto-calculated value.
        timeout: Timeout in seconds for worker processes. Defaults to DEFAULT_WORKER_TIMEOUT.
        show_progress: Whether to show progress (prints to console).
        handle_errors: How to handle errors. Options:
            - "raise": Raise exception on first error (default)
            - "skip": Skip failed items and continue
            - "collect": Collect errors and return them with results
        sequential: If True, force sequential execution in the current process.
    
    Returns:
        List of results in the same order as items.
        If handle_errors="collect", returns list of tuples: (result, error_message)
    
    Example:
        >>> from functools import partial
        >>> def process_item(item, multiplier=2):
        ...     return item * multiplier
        >>> items = [1, 2, 3, 4, 5]
        >>> func = partial(process_item, multiplier=3)
        >>> results = parallel_map(func, items)
        >>> print(results)
        [3, 6, 9, 12, 15]
    
    Raises:
        ValueError: If items list is empty or invalid parameters
        Exception: If handle_errors="raise" and a worker fails
    """
    if not items:
        return []

    # Check for sequential execution request
    if sequential:
        if show_progress:
            print(f"Processing {len(items)} items sequentially (forced)...")
        results = []
        for i, item in enumerate(items):
            if show_progress and len(items) > 1 and (i + 1) % max(1, len(items) // 10) == 0:
                print(f"  Progress: {i+1}/{len(items)}")
            
            result, error = _worker_wrapper(func, item)
            
            if error:
                if handle_errors == "raise":
                    raise RuntimeError(f"Error processing item {i}: {error}")
                elif handle_errors == "collect":
                    results.append((result, error))
                # If "skip", do nothing
            else:
                if handle_errors == "collect":
                    results.append((result, None))
                else:
                    results.append(result)
        return results
    
    # Automatically determine optimal number of workers
    num_workers = get_optimal_worker_count(len(items), task_type=task_type)
    
    # Use single process if only one item or one worker
    if len(items) == 1 or num_workers == 1:
        if show_progress:
            print(f"Processing {len(items)} items sequentially...")
        results = []
        for i, item in enumerate(items):
            if show_progress and len(items) > 1:
                print(f"  Progress: {i+1}/{len(items)}")
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
        chunk_size = max(1, len(items) // (num_workers * 4))
        chunk_size = min(chunk_size, DEFAULT_CHUNK_SIZE)
    
    if timeout is None:
        timeout = DEFAULT_WORKER_TIMEOUT
    
    if show_progress:
        print(f"Processing {len(items)} items with {num_workers} workers (chunk_size={chunk_size})...")
    
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
        for i, (result, error) in enumerate(pool.imap(worker_func, items, chunksize=chunk_size)):
            if show_progress and (i + 1) % max(1, len(items) // 10) == 0:
                print(f"  Progress: {i+1}/{len(items)}")
            
            if error:
                errors_occurred = True
                if handle_errors == "raise":
                    pool.terminate()
                    raise RuntimeError(f"Error processing item {i}: {error}")
                elif handle_errors == "collect":
                    results.append((result, error))
                # If "skip", just don't append anything
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
            print(f"Completed with some errors. Processed {len(results)}/{len(items)} items.")
        else:
            print(f"Completed processing {len(results)} items.")
    
    return results


def parallel_starmap(
    func: Callable,
    args_list: List[Tuple],
    num_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    timeout: Optional[int] = None,
    show_progress: bool = SHOW_MULTIPROCESS_PROGRESS,
) -> List[Any]:
    """
    Execute a function with multiple argument tuples in parallel.
    
    Similar to parallel_map but each item in args_list is a tuple of arguments
    that will be unpacked when calling func.
    
    Args:
        func: Function to execute.
        args_list: List of argument tuples for func.
        num_workers: Number of worker processes.
        chunk_size: Size of chunks for imap.
        timeout: Timeout in seconds for worker processes.
        show_progress: Whether to show progress.
    
    Returns:
        List of results in the same order as args_list.
    
    Example:
        >>> def add(a, b):
        ...     return a + b
        >>> args_list = [(1, 2), (3, 4), (5, 6)]
        >>> results = parallel_starmap(add, args_list)
        >>> print(results)
        [3, 7, 11]
    """
    if not args_list:
        return []
    
    # Wrapper to unpack tuple arguments
    def wrapper(args_tuple):
        return func(*args_tuple)
    
    return parallel_map(
        wrapper,
        args_list,
        num_workers=num_workers,
        chunk_size=chunk_size,
        timeout=timeout,
        show_progress=show_progress,
    )


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

