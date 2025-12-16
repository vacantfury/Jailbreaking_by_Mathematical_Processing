1. Rename input_and_output package and correctly deal with new paths and imports.
1. Finish the experiments (more non_llm processing methods, llm_processor, more evaluator models) and sumbit jobs to the cluster. LLama3 will be fast on the cluster.
1. Find the risk ranks of the promtps in the paper and the competition. Try to treat different ranks of prompts by different approach.
1. Change experiment not to output analysis files.
1. Check that experiment.math_prompt_experiment.MathPromptExperiment takes the right parameters.
1. Refactor LLM_processor classes so that each LLMProcessor class can provide a collection of models services.
1. Complete the experiment of prompt processing by all LLMs using LLM processor list. 
It should be a method of LLM_processor. I updated codes about LLM processor instance list by AI agent but have not checked the correctness of them.
2. Use llama to do set theory or other kind of reformating of input prompts.
2. Try approaches based on probability theory or statistics.
2. Continue to optimize the experiment module. It is too slow now.
3. Explore different LLMs (currently Llama 3) in the current evaluation pipeline
as well as the helper model to generate a score matrix for each LLM_based approach.
3. Try to add more evaluation methods into this project.
4. If to optimize sth (parameters or prompts), 
split datasets like common ml algorithms (k-fold validation? and test set) 
and avoid data leakage.
2. Adapt jailbreak_track.ipynb so that it can read input file with empty line.
(from hugging face? using long chain(?) ?) to replace GPT api.
3. How should I determine the "system" value in gpt message?
2. In MathPromptProcessor, develop several math processing methods.
   The final process() method will be a linear combination of them,
   where the coefficients can be trained out.