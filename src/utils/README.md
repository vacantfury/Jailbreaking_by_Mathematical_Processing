# Utils Module

Utility functions for the project, including multiprocessing utilities optimized for M4 chip with 48GB RAM.

## Features

- **`parallel_map`**: Execute a function on a list of items in parallel
- **`parallel_starmap`**: Execute a function with multiple argument tuples in parallel
- **Error Handling**: Flexible error handling strategies (raise, skip, collect)
- **Progress Tracking**: Optional progress reporting
- **Optimized Configuration**: Tuned for M4 chip performance

## Quick Start

### Basic Parallel Execution

```python
from src.utils import parallel_map

def process_item(item, multiplier=2):
    return item * multiplier

items = [1, 2, 3, 4, 5]
results = parallel_map(process_item, items, multiplier=3)
print(results)  # [3, 6, 9, 12, 15]
```

### With Your Experiment Class

```python
from src.utils import parallel_map
from src.evaluation import Evaluator
from src.llm_utils import LLMModel

class Experiment:
    def __init__(self):
        self.evaluator = Evaluator(model=LLMModel.GPT_4)
    
    def evaluate_single(self, data):
        """Evaluate one prompt-response pair."""
        prompt, risk, response = data
        return self.evaluator.evaluate_harm(prompt, risk, response)
    
    def evaluate_all_parallel(self, test_data):
        """Evaluate all pairs in parallel."""
        # test_data is list of (prompt, risk, response) tuples
        return parallel_map(
            self.evaluate_single,
            test_data,
            num_workers=8  # Utilize M4's cores
        )
```

## API Reference

### `parallel_map`

Execute a function on a list of items in parallel.

```python
parallel_map(
    func: Callable,
    items: List[Any],
    *args,
    num_workers: Optional[int] = None,  # Default: CPU_COUNT - 2
    chunk_size: Optional[int] = None,   # Default: auto-calculated
    timeout: Optional[int] = None,       # Default: 3600 seconds
    show_progress: bool = True,
    handle_errors: str = "raise",        # "raise", "skip", or "collect"
    **kwargs
) -> List[Any]
```

**Parameters:**
- `func`: Function to execute (takes item as first arg)
- `items`: List of items to process
- `*args`: Additional positional arguments for func
- `num_workers`: Number of worker processes (default: optimal for M4)
- `chunk_size`: Size of work chunks per worker
- `timeout`: Max time for worker processes
- `show_progress`: Whether to print progress
- `handle_errors`: How to handle errors:
  - `"raise"`: Stop on first error (default)
  - `"skip"`: Skip failed items, continue processing
  - `"collect"`: Return (result, error) tuples
- `**kwargs`: Additional keyword arguments for func

**Returns:**
- List of results (same order as input)
- If `handle_errors="collect"`: List of `(result, error_message)` tuples

### `parallel_starmap`

Execute a function with multiple argument tuples in parallel.

```python
parallel_starmap(
    func: Callable,
    args_list: List[Tuple],
    num_workers: Optional[int] = None,
    chunk_size: Optional[int] = None,
    timeout: Optional[int] = None,
    show_progress: bool = True,
) -> List[Any]
```

**Example:**
```python
def add(a, b, c):
    return a + b + c

args_list = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
results = parallel_starmap(add, args_list)
# Results: [6, 15, 24]
```

### `get_optimal_worker_count`

Calculate optimal number of workers for your task.

```python
get_optimal_worker_count(
    num_items: int,
    task_type: str = "cpu",  # "cpu" or "io"
    max_workers: Optional[int] = None
) -> int
```

**Example:**
```python
# CPU-bound task (e.g., data processing)
workers = get_optimal_worker_count(100, task_type="cpu")

# I/O-bound task (e.g., API calls)
workers = get_optimal_worker_count(100, task_type="io")
```

## Configuration (constants.py)

### For M4 Chip with 48GB RAM

```python
DEFAULT_NUM_WORKERS = CPU_COUNT - 2  # Reserve 2 cores for OS
MAX_IO_WORKERS = CPU_COUNT * 2       # For I/O-bound tasks
DEFAULT_CHUNK_SIZE = 10              # Balance overhead vs responsiveness
DEFAULT_WORKER_TIMEOUT = 3600        # 1 hour timeout
MULTIPROCESS_START_METHOD = 'spawn'  # Safe for CUDA/torch
```

## Error Handling Strategies

### 1. Raise (Default)

Stop on first error:

```python
results = parallel_map(func, items, handle_errors="raise")
# Raises exception if any item fails
```

### 2. Skip

Skip failed items, continue processing:

```python
results = parallel_map(func, items, handle_errors="skip")
# Returns only successful results
```

### 3. Collect

Collect both results and errors:

```python
results = parallel_map(func, items, handle_errors="collect")
for result, error in results:
    if error:
        print(f"Error: {error}")
    else:
        print(f"Success: {result}")
```

## Performance Tips

### 1. Choose the Right Worker Count

```python
# CPU-bound: Stick to CPU cores (already default)
results = parallel_map(process_data, items)

# I/O-bound: Can use 2x CPU cores
num_workers = get_optimal_worker_count(len(items), task_type="io")
results = parallel_map(call_api, items, num_workers=num_workers)
```

### 2. Batch Size Matters

```python
# Small items (<1000): Use default
results = parallel_map(func, small_items)

# Large items (>10000): Increase chunk size
results = parallel_map(func, large_items, chunk_size=50)
```

### 3. When to Use Multiprocessing

✅ **Use multiprocessing when:**
- Processing many independent items (>10)
- Each item takes significant time (>0.1s)
- Function is CPU-intensive or I/O-bound

❌ **Don't use multiprocessing when:**
- Very few items (<5)
- Very fast operations (<0.01s per item)
- Functions share complex state
- Memory constraints are tight

## Integration Examples

### With Evaluation Module

```python
from src.evaluation import Evaluator
from src.utils import parallel_map

evaluator = Evaluator(model=LLMModel.LLAMA3_8B)

def evaluate_response(data):
    prompt, risk, response = data
    return evaluator.evaluate_harm(prompt, risk, response)

test_data = [...]  # List of (prompt, risk, response) tuples

# Process in parallel
results = parallel_map(evaluate_response, test_data, num_workers=6)
```

### With LLM Generation

```python
from src.llm_utils import LLMServiceFactory, LLMModel
from src.utils import parallel_map

# Note: For API-based LLMs, use their batch methods instead
# This is useful for local models or custom processing

def generate_response(prompt, model_service):
    result = model_service.batch_generate([(f"gen_{prompt}", prompt)])
    return result[0][1]

prompts = ["prompt1", "prompt2", "prompt3", ...]
service = LLMServiceFactory.create(LLMModel.GPT_4)

# Not recommended for API calls (use service.batch_generate instead)
# But useful for local models with custom processing
```

## Troubleshooting

### Issue: "Process was terminated"

**Solution**: Increase timeout or reduce worker count
```python
results = parallel_map(func, items, timeout=7200, num_workers=4)
```

### Issue: Out of memory

**Solution**: Reduce number of workers
```python
results = parallel_map(func, items, num_workers=2)
```

### Issue: Slow performance

**Solution**: Adjust chunk size
```python
# Try larger chunks
results = parallel_map(func, items, chunk_size=50)
```

### Issue: Can't pickle function

**Solution**: Define function at module level, not inside another function
```python
# ❌ Bad
def main():
    def process(x):
        return x * 2
    parallel_map(process, items)  # Won't work!

# ✅ Good
def process(x):
    return x * 2

def main():
    parallel_map(process, items)  # Works!
```

## See Also

- `example_multiprocessor.py` - Complete working examples
- `constants.py` - Configuration options
- Python's `multiprocessing` module documentation

