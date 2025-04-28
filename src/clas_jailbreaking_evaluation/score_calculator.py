import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import src.clas_jailbreaking_evaluation.constants as constants
from src.clas_jailbreaking_evaluation.utils import get_jailbreak_score, get_stl_score
from huggingface_hub import login
class ScoreCalculator:
    """
    Calculator for jailbreaking evaluation scores that performs the complete pipeline:
    1. Loads original and jailbreak prompts
    2. Sets up the model inference pipeline
    3. Generates responses for each prompt
    4. Calculates jailbreak and stealthiness scores
    5. Computes aggregate scores
    """
    
    def __init__(self, model_id=constants.DEFAULT_MODEL):
        """
        Initialize the score calculator with model
        
        Args:
            model_id (str): ID of the model to use for evaluation
        """
        self._huggingface_login()
        self.model_id = model_id
        self.pipe = None
        self.tokenizer = None
        self.set_up_calculator()

    def _huggingface_login(self):
        """
        Login to HuggingFace
        """
        login(token=constants.HUGGINGFACE_TOKEN)

    def get_prompts(self, file_name):
        """
        Load prompts from a jsonl file
        
        Args:
            file_name (str): Path to the prompts file
            
        Returns:
            list: List of prompts
        """
        prompt_list = []
        try:
            with open(file_name, 'r') as file:
                for line in file:
                    json_record = json.loads(line)
                    prompt_list.append(json_record["prompt"])
            return prompt_list
        except Exception as e:
            print(f"Error loading prompts from {file_name}: {e}")
            return []
    
    def set_up_calculator(self):
        """
        Set up the model inference pipeline
        
        Returns:
            pipeline: HuggingFace pipeline for text generation
        """
        print(f"Loading model: {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        pipe = pipeline("text-generation", model=model, tokenizer=self.tokenizer, max_new_tokens=1024)
        self.pipe = pipe

        


    def calculate_all_scores(self, input_prompts_path = constants.DEFAULT_INPUT_PROMPTS_PATH, output_prompts_path = constants.DEFAULT_OUTPUT_PROMPTS_PATH):
        original_prompt_list = self.get_prompts(input_prompts_path)
        jailbreak_prompt_list = self.get_prompts(output_prompts_path)
        avg_jailbreak_score = 0
        avg_stealthiness_score = 0
        for original_prompt, jailbreak_prompt in tqdm(zip(original_prompt_list, jailbreak_prompt_list)):
            messages = [
                {"role": "user", "content": jailbreak_prompt},
            ]
            response = self.pipe(messages)[0]['generated_text'][1]['content']
            jailbreak_score = get_jailbreak_score(self.pipe, original_prompt, jailbreak_prompt, response, self.tokenizer) # In the test phase, the evaluation model and evaluation prompts are held out to avoid adaptive attack on the judge model. This is just an example of model judge. We will consistently use Gemma-2b-it tokenizer to detect jailbreak prompt with length over 100.
            stl_score = get_stl_score(original_prompt, jailbreak_prompt)
            avg_jailbreak_score += jailbreak_score
            avg_stealthiness_score += stl_score
        avg_jailbreak_score /= len(jailbreak_prompt_list)
        avg_stealthiness_score /= len(jailbreak_prompt_list)
        avg_final_score = constants.JAILBREAK_WEIGHT * avg_jailbreak_score + constants.STEALTHINESS_WEIGHT * avg_stealthiness_score
        score_dict = {
            "jailbreak_score": avg_jailbreak_score,
            "stealthiness_score": avg_stealthiness_score,
            "final_score": avg_final_score
        }
        return score_dict
    
    