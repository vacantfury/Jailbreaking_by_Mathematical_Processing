1. The data comes from the 2 repos DRA and starter-kit. Need to introduce them in the article and reports.
1. Found that in splitting and reassembling questions into equations,
the methods that avoids splitting words performs the best.
The score for 6-9 parts of this method is 0.528 for the first 10 iput prompts.
2. But the score for the mode_2 method shows a tendency of increase
up to 9 parts. It is recommended to try more than 9 parts for mode_2 method (and maybe mode_0).
3. For evaluation tokenizer models, gemma can be used well, llama3 can be used directly but slow, diberta needs to change tokenizer signature (not tried).
4. Different evaluator models generate different scores. Gemma 2 only output 0 and 0.5 (guess there will be 1?). 