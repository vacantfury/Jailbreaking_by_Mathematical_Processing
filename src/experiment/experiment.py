"""
Experiment class for managing and executing multiple experiment tasks.

An Experiment can contain multiple Tasks, each with their own configuration.
Tasks are executed in order (queue-based execution).
"""
from typing import List, Dict, Any
from collections import deque

from src.utils.logger import get_logger
from .task import Task

logger = get_logger(__name__)



def _execute_task_safe(task_data):
    """
    Helper function to execute a task safely, catching exceptions.
    Must be at module level for multiprocessing pickling.
    
    Args:
        task_data: Tuple or dict containing task configuration.
    """
    try:
        # If input is a dict (config), recreate Task
        if isinstance(task_data, dict):
             # Import here to avoid circular imports at top level if any
            from src.experiment.task import Task
            task = Task(**task_data)
        else:
            task = task_data

        logger.info(f"Starting task: {task.name}")
        return task.run_task()
    except Exception as e:
        # Check if it's a FatalModelError (by name or import to avoid circular dependency issues if possible)
        # But easier to just import it inside the function or file
        from src.utils.exceptions import FatalModelError
        if isinstance(e, FatalModelError):
            logger.critical(f"FATAL ERROR in task {getattr(task_data, 'name', 'unknown')}: {e}")
            raise # Re-raise to terminate experiment
            
        task_name = task_data.get('name', 'unknown') if isinstance(task_data, dict) else getattr(task_data, 'name', 'unknown')
        logger.error(f"Error executing task '{task_name}': {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'task_name': task_name,
            'error': str(e),
            'status': 'failed'
        }


class Experiment:
    """
    Experiment class for managing multiple experiment tasks.
    
    An Experiment orchestrates multiple Tasks, each representing a unit experiment
    with specific configuration (input data, processing strategy, target model, etc.).
    
    Tasks can be added to the front or tail of the execution queue.
    """
    
    def __init__(self):
        """Initialize the experiment."""
        self.tasks: deque[Task] = deque()
        self.results: List[Dict[str, Any]] = []
    
    def add_task_to_front(self, task: Task):
        """
        Add a task to the front of the execution queue (will execute first).
        
        Args:
            task: Task instance to add
        """
        self.tasks.appendleft(task)
        logger.info(f"Added task '{task.name}' to front of queue (total: {len(self.tasks)} tasks)")
    
    def add_task_to_tail(self, task: Task):
        """
        Add a task to the tail of the execution queue (will execute last).
        
        Args:
            task: Task instance to add
        """
        self.tasks.append(task)
        logger.info(f"Added task '{task.name}' to tail of queue (total: {len(self.tasks)} tasks)")
    
    def run_experiment(self, num_of_tasks: int = None, parallel_tasks: bool = False) -> List[Dict[str, Any]]:
        """
        Execute tasks in the queue in order.
        
        Args:
            num_of_tasks: Number of tasks to execute. If None, execute all tasks.
            parallel_tasks: If True, execute tasks in parallel using multiprocessing.
                           (Note: Prompt processing within tasks will be sequential).
        
        Returns:
            List of all task results
        """
        if not self.tasks:
            logger.warning("No tasks to execute")
            return []
        
        # Determine how many tasks to run
        if num_of_tasks is None:
            task_count = len(self.tasks)
        else:
            task_count = min(num_of_tasks, len(self.tasks))
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Total tasks to execute: {task_count}")
        if parallel_tasks:
            logger.info("Execution Mode: PARALLEL TASKS (Sequential Prompts)")
        else:
            logger.info("Execution Mode: SEQUENTIAL TASKS")
        logger.info(f"{'='*70}\n")
        
        self.results = []
        
        # Prepare tasks for execution
        tasks_to_run = [self.tasks.popleft() for _ in range(task_count)]
        
        if parallel_tasks:
            from src.utils.multiprocessor import parallel_map
            
            # Execute tasks in parallel
            # We use handle_errors="raise" to support fatal termination
            # But specific task errors are swallowed by _execute_task_safe unless they are FatalModelError
            self.results = parallel_map(
                _execute_task_safe,
                tasks_to_run,
                task_type="cpu",
                show_progress=True,
                handle_errors="raise"  # Terminate on FatalModelError
            )
            
            # Unpack results if handle_errors was 'collect'
            # parallel_map returns [(result, error), ...] when collect is used
            # But strict implementation might return list of results if error=None
            # Let's check the implementation logic or simplify.
            # safe parallel_map as implemented in multiprocessor:
            # if handle_errors="collect", returns (result, error)
            
            # Re-process results to match expected format
            final_results = []
            for item in self.results:
                if isinstance(item, tuple) and len(item) == 2:
                    res, err = item
                    if res:
                        final_results.append(res)
                    elif err:
                        # Error caught by worker wrapper but not handled by _execute_task_safe?
                        # actually _execute_task_safe catches exceptions and returns a dict
                        # so usually err is None unless _execute_task_safe failed to catch
                        logger.error(f"Task failed with error: {err}")
                else:
                    final_results.append(item)
            self.results = final_results

        else:
            # Execute tasks in order (Sequential)
            for i, task in enumerate(tasks_to_run):
                logger.info(f"\n{'~'*70}")
                logger.info(f"Executing Task {i+1}/{task_count}: {task.name}")
                logger.info(f"{'~'*70}\n")
                
                try:
                    task_result = task.run_task()
                    self.results.append(task_result)
                except Exception as e:
                    logger.error(f"Error executing task '{task.name}': {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    # Store error result
                    self.results.append({
                        'task_name': task.name,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        # Print overall summary
        self._print_experiment_summary()
        
        return self.results
    
    def _print_experiment_summary(self):
        """Print overall experiment summary."""
        logger.info(f"\n{'='*70}")
        logger.info(f"EXPERIMENT SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total tasks executed: {len(self.results)}")
        
        # Count successes and failures
        successful = sum(1 for r in self.results if r.get('status') != 'failed')
        failed = len(self.results) - successful
        
        logger.info(f"Successful: {successful}")
        if failed > 0:
            logger.info(f"Failed: {failed}")
        
        # Aggregate statistics
        total_prompts = sum(r.get('num_prompts', 0) for r in self.results if r.get('num_prompts'))
        logger.info(f"Total prompts processed: {total_prompts}")
        
        # Calculate average obedience across all tasks
        all_obedience_scores = []
        for task_result in self.results:
            if task_result.get('results'):
                for r in task_result['results']:
                    if r.get('obedience_evaluation') and r['obedience_evaluation'].get('obedience_score') is not None:
                        all_obedience_scores.append(r['obedience_evaluation']['obedience_score'])
        
        if all_obedience_scores:
            avg_overall_obedience = sum(all_obedience_scores) / len(all_obedience_scores)
            logger.info(f"Overall Average Obedience Score: {avg_overall_obedience:.3f}")
        
        logger.info(f"{'='*70}\n")
    
    def get_task_count(self) -> int:
        """Get the number of tasks in the queue."""
        return len(self.tasks)
    
    def clear_tasks(self):
        """Clear all tasks from the queue."""
        self.tasks.clear()
        logger.info("Cleared all tasks from queue")
    
    def __repr__(self) -> str:
        return f"Experiment(tasks={len(self.tasks)})"

