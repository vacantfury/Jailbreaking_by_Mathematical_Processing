
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import src  # noqa: F401
from src.experiment import Task, Experiment
from src.llm_utils import LLMModel
from src.prompt_processor import ProcessorType
from src.utils.logger import get_logger

logger = get_logger(__name__)

def functionality_test():
    """
    Run a small FUNCTIONALITY TEST experiment (2 Tasks, 2 Prompts each).
    Required by user request.
    """
    print("="*80)
    print("FUNCTIONALITY TEST EXPERIMENT (2 Tasks, 2 Prompts)")
    print("="*80)
    
    experiment = Experiment()
    
    # Configuration
    PROMPT_IDS = (0, 2)  # Small subset for testing
    EVALUATION_MODEL = LLMModel.GPT_5_NANO
    STANDARD_MODEL = LLMModel.GPT_4O
    
    # Helper
    def add(name, strategy, target):
        experiment.add_task_to_tail(Task(
            name=name,
            processing_strategy=strategy,
            target_model=target,
            evaluation_model=EVALUATION_MODEL,
            prompt_ids=PROMPT_IDS,
            processing_model=STANDARD_MODEL
        ))

    # Add 2 test tasks
    add("Test_SetTheory", ProcessorType.LLM_SET_THEORY, STANDARD_MODEL)
    add("Test_Markov", ProcessorType.LLM_MARKOV_CHAIN, STANDARD_MODEL)
    
    # Execution
    print(f"\n🚀 Launching Functionality Test...")
    try:
        results = experiment.run_experiment(parallel_tasks=True)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user.")
        return

    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    for r in results:
        score = r.get('statistics', {}).get('jailbreak', {}).get('average_obedience_score', 0.0)
        print(f"Task: {r.get('task_name')} | Score: {score:.3f}")


def experiment_test():
    """
    Run a SCALED comparative experiment (11 Tasks, 50 Prompts each).
    
    Coverage:
    1. Processors: Set, Markov, Quantum, CondProb
    2. Pipelines: Rephrase+Set, Rephrase+Markov
    3. Models: OpenAI (Old/Std/New), Google (Old/New), Anthropic (Std)
    
    Est. Duration: ~15-20 minutes (Parallel Execution)
    """
    print("="*80)
    print("SCALED COMPARATIVE EXPERIMENT (11 Tasks, 50 Prompts)")
    print("="*80)
    
    experiment = Experiment()
    
    # Configuration
    PROMPT_IDS = (0, 49)  # 50 prompts per task!
    EVALUATION_MODEL = LLMModel.GPT_5_NANO
    STANDARD_MODEL = LLMModel.GPT_4O
    
    # Helper
    def add(name, strategy, target, proc_model=LLMModel.GPT_4O):
        experiment.add_task_to_tail(Task(
            name=name,
            processing_strategy=strategy,
            target_model=target,
            evaluation_model=EVALUATION_MODEL,
            prompt_ids=PROMPT_IDS,
            processing_model=proc_model
        ))

    # =========================================================================
    # 1. PROCESSOR COMPARISON (GPT-4o)
    # =========================================================================
    add("1_Proc_SetTheory",   ProcessorType.LLM_SET_THEORY,       STANDARD_MODEL)
    add("2_Proc_Markov",      ProcessorType.LLM_MARKOV_CHAIN,     STANDARD_MODEL)
    add("3_Proc_Quantum",     ProcessorType.LLM_QUANTUM_MECHANICS, STANDARD_MODEL)
    add("4_Proc_CondProb",    ProcessorType.NON_LLM_CONDITIONAL_PROBABILITY, STANDARD_MODEL) # Added
    
    # =========================================================================
    # 2. PIPELINE COMPARISON (GPT-4o)
    # =========================================================================
    add("5_Pipe_RephraseSet",    [ProcessorType.LLM_REPHRASE, ProcessorType.LLM_SET_THEORY], STANDARD_MODEL)
    add("6_Pipe_RephraseMarkov", [ProcessorType.LLM_REPHRASE, ProcessorType.LLM_MARKOV_CHAIN], STANDARD_MODEL) # Added
    
    # =========================================================================
    # 3. MODEL ROBUSTNESS (Processor: Set Theory)
    # =========================================================================
    
    # OpenAI
    add("7_Model_GPT3.5_Old", ProcessorType.LLM_SET_THEORY, LLMModel.GPT_3_5_TURBO)
    # Task 1 is GPT-4o (Standard)
    add("8_Model_GPT5_New",   ProcessorType.LLM_SET_THEORY, LLMModel.GPT_5_NANO)
    
    # Google
    add("9_Model_Gemini1.5_Old", ProcessorType.LLM_SET_THEORY, LLMModel.GEMINI_1_5_FLASH)
    add("10_Model_Gemini2.5_New", ProcessorType.LLM_SET_THEORY, LLMModel.GEMINI_2_5_FLASH_LITE)
    
    # Anthropic (Competitive Benchmark - Substitute)
    add("11_Model_Claude3.7_New", ProcessorType.LLM_SET_THEORY, LLMModel.CLAUDE_3_7_SONNET)

    # =========================================================================
    # 4. QUANTUM MECHANICS EXPANSTION (Processor: Quantum)
    # =========================================================================
    
    # Anthropic (Substitute)
    add("12_Quantum_Claude3.7", ProcessorType.LLM_QUANTUM_MECHANICS, LLMModel.CLAUDE_3_7_SONNET)
    
    # Google
    add("13_Quantum_Gemini1.5Pro", ProcessorType.LLM_QUANTUM_MECHANICS, LLMModel.GEMINI_1_5_PRO)
    add("14_Quantum_Gemini2.5Flash", ProcessorType.LLM_QUANTUM_MECHANICS, LLMModel.GEMINI_2_5_FLASH)
    
    # OpenAI (Comparative)
    add("15_Quantum_GPT3.5", ProcessorType.LLM_QUANTUM_MECHANICS, LLMModel.GPT_3_5_TURBO)
    add("16_Quantum_GPT5Nano", ProcessorType.LLM_QUANTUM_MECHANICS, LLMModel.GPT_5_NANO)
    add("20_Quantum_GPT4Nano", ProcessorType.LLM_QUANTUM_MECHANICS, LLMModel.GPT_4_NANO)

    # =========================================================================
    # 5. SET THEORY EXPANSTION (Processor: Set Theory)
    # =========================================================================
    
    # Anthropic (Substitute)
    add("17_Set_Claude3.7", ProcessorType.LLM_SET_THEORY, LLMModel.CLAUDE_3_7_SONNET)
    
    # Google
    add("18_Set_Gemini1.5Pro", ProcessorType.LLM_SET_THEORY, LLMModel.GEMINI_1_5_PRO)
    add("19_Set_Gemini2.0Flash", ProcessorType.LLM_SET_THEORY, LLMModel.GEMINI_2_0_FLASH)

    # =========================================================================
    # EXECUTION
    # =========================================================================
    count = experiment.get_task_count()
    print(f"\n🚀 Launching {count} Parallel Tasks x 50 Prompts...")
    print("   Total prompts to process: ~1000")
    print("   Estimated time: 30-35 minutes")
    
    try:
        results = experiment.run_experiment(parallel_tasks=True)
    except KeyboardInterrupt:
        print("\n🛑 Experiment interrupted by user.")
        return

    # =========================================================================
    # ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("📊 RESULTS SUMMARY (Average scores)")
    print("="*80)
    
    res_map = {r['task_name']: r for r in results}
    def get_score(key):
        # Look for 'jailbreak' -> 'average_obedience_score' because we standardized it!
        return res_map.get(key, {}).get('statistics', {}).get('jailbreak', {}).get('average_obedience_score', 0.0)

    print("\n1. REFERENCE (Baseline):")
    print(f"   Set Theory (GPT-4o): {get_score('1_Ref_SetTheory_GPT4o'):.3f}")

    print("\n2. NOVEL CONTRIBUTIONS (Our Method):")
    print(f"   Quantum (GPT-4o):    {get_score('2_Novel_Quantum_GPT4o'):.3f}")
    print(f"   Markov (GPT-4o):     {get_score('3_Novel_Markov_GPT4o'):.3f}")

    print("\n3. PIPELINE ABLATION:")
    print(f"   Rephrase + Quantum:  {get_score('4_Pipe_RephraseQuantum_GPT4o'):.3f}")

    print("\n4. DEFENSE ANALYSIS (The Shield):")
    print(f"   Quantum vs Nano:     {get_score('6_Defense_Quantum_GPT5Nano'):.3f}")
    print(f"   Set vs Nano:         {get_score('8_Defense_Set_GPT5Nano'):.3f}")

    print("\n5. ROBUSTNESS (Google Family):")
    print(f"   Gemini 3 Flash:      {get_score('7_Robust_Quantum_Gemini3Flash'):.3f}")
    print(f"   Gemini 2.0 Lite:     {get_score('9_Robust_Quantum_Gemini20Lite'):.3f}")

    print("\n6. NEGATIVE CONTROL:")
    print(f"   Cond. Prob (Failure): {get_score('10_Control_CondProb_GPT4o'):.3f}")

def test_refactoring():
    """
    Test the REFACTORED system:
    1. Dataset loading (harmbench_small)
    2. Baseline Mode
    3. Baseline Reuse
    4. HarmBench Evaluation
    """
    print("="*80)
    print("REFACTORING VERIFICATION TEST")
    print("="*80)
    
    experiment = Experiment()
    
    # Common Config
    IDS = (0, 1) # Test just 2 prompts
    TARGET = LLMModel.GPT_3_5_TURBO # Cheap model
    EVAL_MODEL = LLMModel.GPT_4O
    
    # 1. Baseline Run (Youjia Method)
    print("\n🔹 Step 1: Running Baseline Mode (Youjia)...")
    task_baseline = Task(
        name="refactor_baseline",
        processing_strategy=ProcessorType.LLM_REPHRASE, # Ignored in baseline mode
        target_model=TARGET,
        evaluation_model=EVAL_MODEL,
        evaluation_method="youjia",
        mode="baseline",
        input_data_path="harmbench_small",
        prompt_ids=IDS
    )
    experiment.add_task_to_tail(task_baseline)
    results = experiment.run_experiment()
    baseline_dir = results[0]['experiment_dir']
    print(f"   Baseline saved to: {baseline_dir}")
    
    # 2. Reuse Run (Youjia Method)
    print("\n🔹 Step 2: Running Normal Mode with Baseline Reuse...")
    experiment = Experiment() # Reset
    task_reuse = Task(
        name="refactor_reuse",
        processing_strategy=ProcessorType.LLM_REPHRASE,
        target_model=TARGET,
        evaluation_model=EVAL_MODEL,
        evaluation_method="youjia",
        mode="normal",
        baseline_result_dir=baseline_dir, # REUSE
        input_data_path="harmbench_small",
        prompt_ids=IDS,
        processing_model=LLMModel.GPT_3_5_TURBO
    )
    experiment.add_task_to_tail(task_reuse)
    experiment.run_experiment()
    
    # 3. HarmBench Evaluation Run
    print("\n🔹 Step 3: Running HarmBench Evaluation...")
    experiment = Experiment() # Reset
    task_hb = Task(
        name="refactor_harmbench",
        processing_strategy=ProcessorType.LLM_REPHRASE,
        target_model=TARGET,
        evaluation_model=EVAL_MODEL,
        evaluation_method="harmbench", # NEW METHOD
        mode="normal",
        input_data_path="harmbench_small",
        prompt_ids=IDS,
        processing_model=LLMModel.GPT_3_5_TURBO
    )
    experiment.add_task_to_tail(task_hb)
    experiment.run_experiment()
    
    print("\n✅ Refactoring Test Complete!")


def run_full_scale_experiments():
    """
    Run the FULL SCALE experiments (H1, H2, Novelty, & Pipeline).
    Total Tasks: 8
    Estimated Duration: ~30 mins on M4 Max (Parallel).
    """
    print("="*80)
    print("FULL SCALE EXPERIMENT: HARMBENCH EVALUATION")
    print("="*80)
    
    experiment = Experiment()
    
    # -------------------------------------------------------------------------
    # 1. REFERENCE (Set Theory - [Bethany2024])
    # -------------------------------------------------------------------------
    # "The Hammer" - Testing reproducibility of previous work on HarmBench
    experiment.add_task_to_tail(Task(
        name="1_Ref_SetTheory_GPT4o",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",  # Fix: Explicitly load correct dataset (250 items)
        processing_strategy=ProcessorType.LLM_SET_THEORY,
        target_model=LLMModel.GPT_4O,
        evaluation_method="harmbench"
    ))

    # -------------------------------------------------------------------------
    # 2. OUR NOVEL CONTRIBUTIONS (Quantum & Markov)
    # -------------------------------------------------------------------------
    # H1 Generalization: Does the principle work in other domains?
    experiment.add_task_to_tail(Task(
        name="2_Novel_Quantum_GPT4o",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=ProcessorType.LLM_QUANTUM_MECHANICS,
        target_model=LLMModel.GPT_4O,
        evaluation_method="harmbench"
    ))
    
    experiment.add_task_to_tail(Task(
        name="3_Novel_Markov_GPT4o",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=ProcessorType.LLM_MARKOV_CHAIN,
        target_model=LLMModel.GPT_4O,
        evaluation_method="harmbench"
    ))

    # -------------------------------------------------------------------------
    # 3. PIPELINE ABLATION (Pre-processing Analysis)
    # -------------------------------------------------------------------------
    # Does adding a "Rephrase" step improve our Novel Method (Quantum)?
    experiment.add_task_to_tail(Task(
        name="4_Pipe_RephraseQuantum_GPT4o",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=[ProcessorType.LLM_REPHRASE, ProcessorType.LLM_QUANTUM_MECHANICS],
        target_model=LLMModel.GPT_4O,
        evaluation_method="harmbench"
    ))

    # -------------------------------------------------------------------------
    # 4. DEFENSE ANALYSIS (H2: Active Filtering) - USING NOVEL METHOD
    # -------------------------------------------------------------------------
    # The "Shield Test" - Testing H2 using OUR NEW method (Quantum)
    experiment.add_task_to_tail(Task(
        name="6_Defense_Quantum_GPT5Nano",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=ProcessorType.LLM_QUANTUM_MECHANICS,
        target_model=LLMModel.GPT_5_NANO,
        evaluation_method="harmbench"
    ))

    # -------------------------------------------------------------------------
    # 5. ROBUSTNESS (Cross-Model Validation) - USING NOVEL METHOD
    # -------------------------------------------------------------------------
    # Google Family - Testing OUR NEW method (Quantum) on Google (Next Gen)
    experiment.add_task_to_tail(Task(
        name="7_Robust_Quantum_Gemini3Flash",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=ProcessorType.LLM_QUANTUM_MECHANICS,
        target_model=LLMModel.GEMINI_3_FLASH_PREVIEW,
        evaluation_method="harmbench"
    ))
    
    # -------------------------------------------------------------------------
    # 5. EXPANDED ROBUSTNESS & DEFENSE
    # -------------------------------------------------------------------------
    # Comparison: Does GPT-5-Nano block usage of Set Theory too? (Baseline Defense)
    experiment.add_task_to_tail(Task(
        name="8_Defense_Set_GPT5Nano",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=ProcessorType.LLM_SET_THEORY,
        target_model=LLMModel.GPT_5_NANO,
        evaluation_method="harmbench"
    ))

    # New Model: Testing Quantum on Gemini 2.0 Flash Lite (Legacy Lite)
    experiment.add_task_to_tail(Task(
        name="9_Robust_Quantum_Gemini20Lite",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=ProcessorType.LLM_QUANTUM_MECHANICS,
        target_model=LLMModel.GEMINI_2_0_FLASH_LITE,
        evaluation_method="harmbench"
    ))

    # -------------------------------------------------------------------------
    # 6. NEGATIVE CONTROL (Low Complexity)
    # -------------------------------------------------------------------------
    # "The Failure Case" - Testing if simple math fails (Validation of Complexity Hypothesis)
    experiment.add_task_to_tail(Task(
        name="10_Control_CondProb_GPT4o",
        dataset_name="harmbench_large",
        input_data_path="harmbench_large",
        processing_strategy=ProcessorType.NON_LLM_CONDITIONAL_PROBABILITY,
        target_model=LLMModel.GPT_4O,
        evaluation_method="harmbench"
    ))

    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    print(f"\n🚀 Launching {experiment.get_task_count()} Parallel Tasks...")
    print("   Target: HarmBench Large (250 prompts)")
    print("   Judge:  GPT-4o (Standard)")
    
    # Run tasks in parallel to speed up execution
    try:
        experiment.run_experiment(parallel_tasks=True)
    except KeyboardInterrupt:
        print("\n🛑 Experiment interrupted by user.")
        return

if __name__ == "__main__":
    # Uncomment the one you want to run
    # test_refactoring()
    # functionality_test()
    # experiment_test()
    run_full_scale_experiments() # Default to running the full scale as requested

