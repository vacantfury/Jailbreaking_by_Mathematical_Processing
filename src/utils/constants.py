"""
Constants for utility functions.
"""
import multiprocessing
import os

# ============================================================================
# Multiprocessing Configuration
# ============================================================================

# Get number of available CPU cores
_CPU_COUNT = multiprocessing.cpu_count()

# For M4 chip with 48GB RAM, we can be aggressive with parallelization
# M4 typically has 10-14 cores (4 performance + 6-10 efficiency)
# Reserve 2 cores for OS and other tasks
DEFAULT_NUM_WORKERS = max(1, _CPU_COUNT - 2)

# Maximum workers for I/O-bound tasks (can exceed CPU count)
MAX_IO_WORKERS = _CPU_COUNT * 2

# Minimum batch size per worker (avoid overhead for tiny batches)
MIN_BATCH_SIZE_PER_WORKER = 1

# Maximum batch size per worker (prevent memory issues)
# With 48GB RAM, we can handle large batches, but be conservative
MAX_BATCH_SIZE_PER_WORKER = 1000

# Chunk size for imap operations (balance between overhead and responsiveness)
DEFAULT_CHUNK_SIZE = 10

# Minimum number of items to trigger parallel processing
# Below this threshold, sequential processing is faster due to overhead
PARALLEL_PROCESSING_THRESHOLD = 10

# Timeout for worker processes (in seconds)
DEFAULT_WORKER_TIMEOUT = 3600  # 1 hour

# Method to start processes ('spawn', 'fork', 'forkserver')
# 'spawn' is safest for complex objects and CUDA, 'fork' is faster on Unix
MULTIPROCESS_START_METHOD = os.getenv('MULTIPROCESS_START_METHOD', 'spawn')

# ============================================================================
# Logging Configuration
# ============================================================================

# Whether to show progress for multiprocessing operations
SHOW_MULTIPROCESS_PROGRESS = False  # Disabled to reduce log clutter

# Log level for multiprocessing operations
MULTIPROCESS_LOG_LEVEL = "INFO"

