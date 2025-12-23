"""
Task - a unit experiment with specific configuration and execution logic.
"""
import json
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from src.llm_utils import LLMModel
from src.evaluation import EvaluatorFactory, load_and_filter_prompts
from src.evaluation.constants import DEFAULT_EVALUATION_MODEL
from src.evaluation import constants as eval_constants
from src.prompt_processor import create_processor, ProcessorType
from src.utils.logger import get_logger
from src.utils import generate_task_filename
from src.experiment.constants import TASK_DATA_DIR, DEFAULT_TASK_NAME, DATASET_NAME_TO_PATH, DATASET_NAME_TO_EXPERIMENT_PATH
from src.data.constants import ATA_HARMFUL_PROMPTS_FILE

# Alias for backward compatibility in this file
HARMFUL_PROMPTS_FILE = ATA_HARMFUL_PROMPTS_FILE

logger = get_logger(__name__)


class Task:
    """
    A Task represents a single unit experiment with specific configuration.
    
    Each task has:
    - Input data path (prompts to process)
    - Output data path (where to save results)
    - Evaluation model (for scoring responses)
    - Processing strategy (how to transform prompts)
    - Processing model (LLM for processing, if applicable)
    """
    
    def __init__(
        self,
        processing_strategy: Union[
            str, 
            ProcessorType, 
            List[Union[str, ProcessorType]],
            List[Tuple[Union[str, ProcessorType], Optional[LLMModel]]]
        ],
        target_model: LLMModel,
        evaluation_model: Optional[LLMModel] = None,
        evaluation_method: str = "youjia",
        mode: str = "normal",
        baseline_result_dir: Optional[Union[str, Path]] = None,
        name: Optional[str] = None,
        input_data_path: Optional[Union[str, Path]] = None,
        experiment_dir: Optional[Path] = None,
        processing_model: Optional[LLMModel] = None,
        prompt_ids: Optional[Union[Tuple[int, int], List[Union[int, str]]]] = None,
        dataset_name: Optional[str] = None,
        **strategy_kwargs
    ):
        """
        Initialize a Task.
        
        Args:
            processing_strategy: Processor(s) to transform prompts.
            target_model: Target LLM to test/jailbreak.
            evaluation_model: LLM model for evaluation.
            evaluation_method: Method to use ('youjia' or 'harmbench'). Default 'youjia'.
            mode: Experiment mode ('normal' or 'baseline'). 
                  'baseline': Only runs target model on original prompts and evaluates.
                  'normal': Runs processing/jailbreak (and baseline if not provided).
            baseline_result_dir: Path to directory containing previous baseline results to reuse.
            name: Name of the task.
            input_data_path: Path to input prompts file or dataset alias.
            experiment_dir: Directory to save task results.
            processing_model: LLM model for processing.
            prompt_ids: Which prompts to test.
            dataset_name: Name of the dataset being used (for directory organization).
            **strategy_kwargs: Additional arguments for the processing strategy(s).
        """
        self.name = name or DEFAULT_TASK_NAME
        self.dataset_name = dataset_name
        
        # Resolve input_data_path and dataset_name
        if input_data_path:
            # Check if it's a known dataset alias first
            input_str = str(input_data_path)
            if input_str in DATASET_NAME_TO_PATH:
                self.input_data_path = DATASET_NAME_TO_PATH[input_str]
                if not self.dataset_name:
                    self.dataset_name = input_str
            else:
                self.input_data_path = Path(input_data_path)
        else:
            # Default to harmful prompts if nothing provided
            self.input_data_path = HARMFUL_PROMPTS_FILE
            if not self.dataset_name:
                self.dataset_name = "ata_harmful"
            
        # Evaluation settings
        self.evaluation_model = evaluation_model if evaluation_model is not None else DEFAULT_EVALUATION_MODEL
        self.evaluation_method = evaluation_method
        self.mode = mode.lower()
        self.baseline_result_dir = Path(baseline_result_dir) if baseline_result_dir else None
        
        # Parse processing_strategy
        self.processing_strategies = []
        self.processor_models = []
        
        if isinstance(processing_strategy, (str, ProcessorType)):
            self.processing_strategies = [processing_strategy]
            self.processor_models = [processing_model]
        else:
            strategy_list = list(processing_strategy)
            for item in strategy_list:
                if isinstance(item, tuple):
                    proc_type, proc_model = item
                    self.processing_strategies.append(proc_type)
                    if proc_model is not None and isinstance(proc_model, LLMModel):
                        self.processor_models.append(proc_model)
                    else:
                        self.processor_models.append(processing_model)
                else:
                    self.processing_strategies.append(item)
                    self.processor_models.append(processing_model)
        
        self.processing_model = processing_model
        self.target_model = target_model
        self.prompt_ids = prompt_ids
        self.strategy_kwargs = strategy_kwargs
        
        # Set up experiment directory path
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
        else:
            # Determine base directory based on dataset
            base_dir = TASK_DATA_DIR
            if self.dataset_name and self.dataset_name in DATASET_NAME_TO_EXPERIMENT_PATH:
                base_dir = DATASET_NAME_TO_EXPERIMENT_PATH[self.dataset_name]
            
            # Create a unique timestamped folder name
            folder_name = generate_task_filename(self.name).replace('.jsonl', '')
            self.experiment_dir = base_dir / folder_name
        
        # Lazy initialization
        self.evaluator = None
        self.processors = None
        
        # Results storage
        self.results: Dict[str, Any] = {}
        
        logger.info(f"Initialized Task '{self.name}' (Mode: {self.mode})")
        logger.info(f"Dataset: {self.dataset_name}, Experiment Dir: {self.experiment_dir}")

    def load_prompts(self) -> list[Dict[str, Any]]:
        """Load prompts from input data path."""
        return load_and_filter_prompts(self.input_data_path, self.prompt_ids)
    
    def _load_baseline_results(self) -> Optional[Dict[str, Any]]:
        """Load baseline results from detailed_results.jsonl in baseline_result_dir."""
        if not self.baseline_result_dir:
            return None
            
        detailed_file = self.baseline_result_dir / "detailed_results.jsonl"
        if not detailed_file.exists():
            logger.warning(f"Baseline results file not found at {detailed_file}")
            return None
            
        try:
            results_list = []
            with open(detailed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results_list.append(json.loads(line))
            
            logger.info(f"Loaded {len(results_list)} baseline results from {self.baseline_result_dir}")
            return {'results': results_list}
        except Exception as e:
            logger.error(f"Failed to load baseline results: {e}")
            return None

    def run_task(self) -> Dict[str, Any]:
        """Execute the task."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Task: {self.name}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"{'='*60}\n")
        
        # Step 1: Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
        # Step 2: Load prompts
        prompts = self.load_prompts()
        if not prompts:
            logger.error("No prompts loaded")
            return {}
        
        logger.info(f"Loaded {len(prompts)} prompts from: {self.input_data_path}")
        
        if not self.target_model:
            logger.error("No target model specified")
            return {}

        # Initialize evaluator
        if self.evaluator is None:
            self.evaluator = EvaluatorFactory.create(
                method=self.evaluation_method,
                model=self.evaluation_model
            )
            
        # Determine baseline results (reuse if normal mode and dir provided)
        baseline_results = None
        if self.mode == "normal" and self.baseline_result_dir:
            baseline_results = self._load_baseline_results()
        
        # Step 3: Process Prompts & Run
        
        if self.mode == "baseline":
            # In baseline mode, we skip processing and just evaluate original prompts
            # We treat original prompts as "processed" prompts so the evaluator runs them
            processed_texts = [p['prompt'] for p in prompts]
            
            # Since processed == original, step 2 (jailbreak) in evaluator will mirror step 1 (baseline)
            # This is fine; we will just save the baseline parts suitable for reuse
            logger.info("Running in BASELINE mode (skipping prompt processing)")
            evaluation_results = self.evaluator.evaluate_jailbreak_effectiveness(
                prompts=prompts,
                processed_prompts=processed_texts,
                target_model=self.target_model
            )
            
            # For result storage, we clear the redundant "jailbreak" data to avoid confusion
            # Actually, let's keep it but mark the strategy as 'baseline'
            processing_pipeline = [("baseline", None)]
            strategy_str = "baseline"
            all_stage_results = []

        else:
            # Normal mode: Run processing pipeline
            if self.processors is None:
                self.processors = []
                for strategy, proc_model in zip(self.processing_strategies, self.processor_models):
                    processor = create_processor(
                        name=strategy,
                        model=proc_model,
                        **self.strategy_kwargs
                    )
                    self.processors.append(processor)
            
            prompt_texts = [p['prompt'] for p in prompts]
            all_stage_results = []
            
            if len(self.processors) == 1:
                logger.info(f"Processing {len(prompt_texts)} prompts with strategy: {self.processing_strategies[0]}")
                processed_texts = self.processors[0].batch_process(prompt_texts)
                all_stage_results.append(processed_texts)
            else:
                processed_texts = prompt_texts
                for i, (processor, strategy) in enumerate(zip(self.processors, self.processing_strategies), 1):
                    logger.info(f"  Stage {i}/{len(self.processors)}: Applying {strategy}")
                    processed_texts = processor.batch_process(processed_texts)
                    all_stage_results.append(processed_texts)
            
            # Run evaluation (passing baseline_results if available)
            logger.info("Running evaluation...")
            evaluation_results = self.evaluator.evaluate_jailbreak_effectiveness(
                prompts=prompts,
                processed_prompts=processed_texts,
                target_model=self.target_model,
                baseline_results=baseline_results
            )
            
            processing_pipeline = [
                (str(proc), model.value if model else None)
                for proc, model in zip(self.processing_strategies, self.processor_models)
            ]
            strategy_str = " -> ".join([proc for proc, _ in processing_pipeline]) if len(processing_pipeline) > 1 else processing_pipeline[0][0]

        # Step 5: Enrich results
        final_results = []
        
        for idx, eval_result in enumerate(evaluation_results['results']):
            # Build list of processed prompts from each stage (for normal mode)
            processed_prompts_list = []
            for stage_results in all_stage_results:
                if idx < len(stage_results):
                    processed_prompts_list.append(stage_results[idx])
            
            final_results.append({
                **eval_result,
                'target_model': self.target_model.value,
                'processing_strategy': strategy_str,
                'processing_pipeline': processing_pipeline,
                'processed_prompts': processed_prompts_list,
            })
        
        self.results = {
            'task_name': self.name,
            'input_path': str(self.input_data_path),
            'experiment_dir': str(self.experiment_dir),
            'processing_strategy': strategy_str,
            'processing_pipeline': processing_pipeline,
            'processing_model': self.processing_model.value if self.processing_model else None,
            'target_model': self.target_model.value,
            'evaluation_model': self.evaluation_model.value,
            'evaluation_method': self.evaluation_method,
            'mode': self.mode,
            'num_prompts': len(prompts),
            'results': final_results,
            'statistics': evaluation_results['statistics'],
        }
        
        self._save_results_to_folder()
        self._print_summary()
        
        return self.results
    
    def _save_results_to_folder(self):
        """Save task results to folder structure."""
        if not self.results:
            return
        
        logger.info(f"Saving results to {self.experiment_dir}")
        config_file = self.experiment_dir / "config.md"
        summary_file = self.experiment_dir / "summary.json"
        detailed_file = self.experiment_dir / "detailed_results.jsonl"
        
        # 1. Detailed Results
        with open(detailed_file, 'w', encoding='utf-8') as f:
            for result in self.results['results']:
                # Ensure we save all keys, especially baseline data
                f.write(json.dumps(result) + '\n')
                
        # 2. Config & Summary
        statistics = self.results.get('statistics', {})
        
        # Basic filtering for valid scores
        def get_valid_scores(key):
             return [r[key] for r in self.results['results'] if r.get(key) is not None]

        baseline_scores = get_valid_scores('baseline_obedience_score')
        jailbreak_scores = get_valid_scores('jailbreak_obedience_score')
        effectiveness_scores = get_valid_scores('jailbreak_effectiveness')
        
        # self._write_config_markdown(config_file, baseline_scores, jailbreak_scores, effectiveness_scores)
        
        summary = {
            'configuration': {
                'task_name': self.results['task_name'],
                'input_path': self.results['input_path'],
                'experiment_dir': self.results['experiment_dir'],
                'processing_strategy': self.results['processing_strategy'],
                'target_model': self.results['target_model'],
                'processing_model': self.results['processing_model'],
                'evaluation_model': self.results['evaluation_model'],
                'mode': self.mode,
                'evaluation_method': self.evaluation_method
            },
            'data': {'num_prompts': self.results['num_prompts']},
            'statistics': statistics
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
            
        logger.info("✅ Saved task results")

    def _write_config_markdown(self, config_file: Path, baseline_scores: list, jailbreak_scores: list, effectiveness_scores: list):
        """Write human-readable configuration and results to Markdown file."""
        import datetime
        
        avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
        avg_jailbreak = sum(jailbreak_scores) / len(jailbreak_scores) if jailbreak_scores else 0
        success_rate = (sum(1 for e in effectiveness_scores if e > 0) / len(effectiveness_scores) * 100) if effectiveness_scores else 0
        
        content = f"""# Task Configuration and Results
**Task Name:** `{self.name}`  
**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Mode:** `{self.mode}`  
**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Mode:** `{self.mode}`  
**Method:** `{self.evaluation_method}`
**Input Path:** `{self.results['input_path']}`

## Strategy
`{self.results['processing_strategy']}`

## Results
- **Baseline Avg:** {avg_baseline:.3f}
- **Jailbreak Avg:** {avg_jailbreak:.3f}
- **Success Rate:** {success_rate:.1f}%
"""
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def _print_summary(self):
        """Print summary."""
        stats = self.results.get('statistics', {})
        logger.info(f"Task Complete. Stats: {stats}")

    def __repr__(self) -> str:
        return f"Task(name='{self.name}', mode='{self.mode}', method='{self.evaluation_method}')"
