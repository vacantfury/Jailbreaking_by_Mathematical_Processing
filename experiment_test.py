
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
    # 5. SET THEORY EXPANSION (Processor: Set Theory)
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
        return res_map.get(key, {}).get('statistics', {}).get('jailbreak', {}).get('average_obedience_score', 0.0)

    print("\n1. PROCESSORS (GPT-4o):")
    print(f"   Set Theory: {get_score('1_Proc_SetTheory'):.3f}")
    print(f"   Markov:     {get_score('2_Proc_Markov'):.3f}")
    print(f"   Quantum:    {get_score('3_Proc_Quantum'):.3f}")
    print(f"   CondProb:   {get_score('4_Proc_CondProb'):.3f}")

    print("\n2. PIPELINE VALUE:")
    print(f"   Reph+Set:   {get_score('5_Pipe_RephraseSet'):.3f} (vs {get_score('1_Proc_SetTheory'):.3f})")
    print(f"   Reph+Mark:  {get_score('6_Pipe_RephraseMarkov'):.3f} (vs {get_score('2_Proc_Markov'):.3f})")
    
    print("\n3. MODEL ROBUSTNESS:")
    print(f"   GPT-3.5:    {get_score('7_Model_GPT3.5_Old'):.3f}")
    print(f"   GPT-5-Nano: {get_score('8_Model_GPT5_New'):.3f}")
    print(f"   Gemini 1.5: {get_score('9_Model_Gemini1.5_Old'):.3f}")
    print(f"   Gemini 2.5: {get_score('10_Model_Gemini2.5_New'):.3f}")
    print(f"   Claude 3.7: {get_score('11_Model_Claude3.7_New'):.3f}")
    
    print("\n4. EXPANDED EXPERIMENTS (Quantum):")
    print(f"   Claude 3.7: {get_score('12_Quantum_Claude3.7'):.3f}")
    print(f"   Gemini 1.5 Pro: {get_score('13_Quantum_Gemini1.5Pro'):.3f}")
    print(f"   Gemini 2.5 Flash: {get_score('14_Quantum_Gemini2.5Flash'):.3f}")
    print(f"   GPT-3.5: {get_score('15_Quantum_GPT3.5'):.3f}")
    print(f"   GPT-5-Nano: {get_score('16_Quantum_GPT5Nano'):.3f}")
    print(f"   GPT-4-Nano: {get_score('20_Quantum_GPT4Nano'):.3f}")

    print("\n5. EXPANDED EXPERIMENTS (Set Theory):")
    print(f"   Claude 3.7: {get_score('17_Set_Claude3.7'):.3f}")
    print(f"   Gemini 1.5 Pro: {get_score('18_Set_Gemini1.5Pro'):.3f}")
    print(f"   Gemini 2.0 Flash: {get_score('19_Set_Gemini2.0Flash'):.3f}")

if __name__ == "__main__":
    functionality_test()
    # experiment_test()
