
import os
import json
import glob
from pathlib import Path

def generate_table():
    base_dir = Path("/Users/haoyu/Files/US study life and job/study and life/NeU/courses/Fall 2025/CS 8674/repos/Jailbreaking_by_Mathematical_Processing/src/data/experiment_data")
    
    results = []
    
    # Iterate through all subdirectories
    # FILTER: Only folders starting with 202512
    for summary_file in glob.glob(str(base_dir / "202512*/summary.json")):
        try:
            with open(summary_file, 'r') as f:
                data = json.load(f)
                
            config = data.get('configuration', {})
            stats = data.get('statistics', {})
            jailbreak_stats = stats.get('jailbreak', {})
            effectiveness = stats.get('effectiveness', {})
            
            task_name = config.get('task_name', 'Unknown')
            # simplify strategy name
            strategy = config.get('processing_strategy', 'Unknown').replace('ProcessorType.', '').replace('LLM_', '').replace('NON_LLM_', '').replace('_CHAIN', '').replace('_MECHANICS', '').replace('_THEORY', '')
            
            target_model = config.get('target_model', ['Unknown'])[0]
            
            # Formatting
            obedience = jailbreak_stats.get('average_obedience_score', 0.0)
            success_rate = effectiveness.get('success_rate', 0.0)
            
            results.append({
                'task': task_name.replace('_', '\\_'), # Escape underscores for LaTeX
                'strategy': strategy.title(),
                'model': target_model,
                'obedience': obedience,
                'success': success_rate
            })
            
        except Exception as e:
            print(f"Error reading {summary_file}: {e}")

    # Sort by Task Name to keep them in order (1, 2, ... 20)
    # Extract number from task name if possible
    def sort_key(x):
        try:
            return int(x['task'].split('\_')[0]) # Note: backslash is already escaped in string
        except:
            return 999
            
    results.sort(key=sort_key)

    print("\\begin{table}[h]")
    print("\\caption{Complete Experimental Data (25 Runs)}")
    print("\\label{tab:all_data}")
    print("\\centering")
    print("\\tiny") # Use tiny font for large table
    print("\\begin{tabular}{l|l|c|c}") # Removed Task ID column
    print("\\toprule")
    print("\\textbf{Strategy} & \\textbf{Target Model} & \\textbf{Success} & \\textbf{Obedience} \\\\")
    print("\\midrule")
    
    for r in results:
        # Removed Task ID from row output
        print(f"{r['strategy']} & {r['model']} & {r['success']}\\% & {r['obedience']:.3f} \\\\")
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    generate_table()
