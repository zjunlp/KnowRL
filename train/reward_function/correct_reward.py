import os
import re
import logging
from typing import List, Optional
from openai import OpenAI
from zhipuai import ZhipuAI
from api_client_manager import ApiClientManager
from .prompt.reward_prompts import LLM_EVAL_PROMPT
from .reward_utils import save_json_output, get_timestamp

logger = logging.getLogger(__name__)


USE_API_MANAGER_FOR_LLM_EVAL = os.environ.get("USE_API_MANAGER_FOR_LLM_EVAL", "True").lower() == "true"
api_manager = ApiClientManager.get_instance()

class RewardEvaluator:
    def __init__(self, api_key: str, api_base: str):
        if USE_API_MANAGER_FOR_LLM_EVAL:
            logger.info("Using API manager to get LLM evaluation client")
            self.client = api_manager.get_client(api_key=api_key, base_url=api_base)
            self.use_api_manager = True
        else:
            logger.info("Using standard method to get LLM evaluation client")
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.use_api_manager = False
            
        self.model_name = "gpt-4o-mini"
        logger.info(f"Using model: {self.model_name}")
        
        
    def reset_refusal_flags(self):
        self.refusal_flags = []

    def get_refusal_flags(self):
        return self.refusal_flags
    
    def get_score(self, question: str, answer: str, best_answer: Optional[str] = None, prompt_template: Optional[str] = None):
        try:
            best_answer = best_answer or "No reference answer provided."
            template = prompt_template or LLM_EVAL_PROMPT
            
            prompt = template.format(
                question=question,
                answer=answer,
                gold_answer=best_answer
            )
            
            if self.use_api_manager:
                def make_request():
                    return self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=4096
                    )
                
                response = api_manager.execute_request(make_request)
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=4096
                )
            
            raw_response_text = response.choices[0].message.content.strip()
            response_text = raw_response_text.upper()
            
            # Regular expressions for key scoring terms
            correct_pattern = r'\b(CORRECT|A)\b'
            incorrect_pattern = r'\b(INCORRECT|B)\b'
            not_attempted_pattern = r'\b(NOT_ATTEMPTED|NOT ATTEMPTED|C)\b'
            
            # First check if there's an <ANSWER> tag
            answer_match = re.search(r"<ANSWER>(.*?)</ANSWER>", response_text, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                
                if re.search(correct_pattern, answer_content):
                    score = 2.0
                elif re.search(incorrect_pattern, answer_content):
                    score = -1.0
                elif re.search(not_attempted_pattern, answer_content):
                    score = 1.0
                else:
                    if re.search(correct_pattern, response_text):
                        score = 2.0
                    elif re.search(incorrect_pattern, response_text):
                        score = -1.0
                    elif re.search(not_attempted_pattern, response_text):
                        score = 1.0
                    else:
                        logger.warning(f"Unrecognized scoring result: '{raw_response_text}'")
                        score = 0.0
            else:
                if re.search(correct_pattern, response_text):
                    score = 2.0
                elif re.search(incorrect_pattern, response_text):
                    score = -1.0
                elif re.search(not_attempted_pattern, response_text):
                    score = 1.0
                else:
                    logger.warning(f"Unrecognized scoring result: '{raw_response_text}'")
                    score = 0.0
            
            return score, raw_response_text
                    
        except Exception as e:
            logger.error(f"LLM scoring error: {str(e)}")
            return 0.0, f"ERROR: {str(e)}"


def llm_eval_reward_func(prompts: List[str], completions: List[str], best_answer: Optional[List[str]] = None, 
                        qwestion: Optional[List[str]] = None, **kwargs) -> List[float]:
    rewards = []
    log_entries = []
    refusal_flags = []  

    judge_llm = RewardEvaluator(
        api_key=os.environ.get("OPENAI_API_KEY_JUDGE"),
        api_base=os.environ.get("OPENAI_API_BASE_JUDGE")
    )

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            # Extract answer part
            match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
            if match is None:
                rewards.append(-1)
                refusal_flags.append(False)  
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "completion": completion,
                    "error": "answer tag not found",
                    "reward": -1,
                    "is_refusal": False,
                    "judge_llm_response": "N/A",
                    "timestamp": get_timestamp()
                })
                continue
            predicted_answer = match.group(1).strip()

            if not predicted_answer or predicted_answer.isspace():
                rewards.append(-1)
                refusal_flags.append(False)  
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "completion": completion,
                    "error": "empty content in answer tag",
                    "reward": -1,
                    "is_refusal": False,
                    "raw_llm_response": "N/A",
                    "timestamp": get_timestamp()
                })
                continue

            best_ans = best_answer[i] if best_answer and i < len(best_answer) else None
            question = qwestion[i] if qwestion and i < len(qwestion) else prompt

            
            score, response_text = judge_llm.get_score(
                question,
                predicted_answer,
                best_ans,
                prompt_template=LLM_EVAL_PROMPT
            )
            
           
            is_refusal = (score == 1.0)
            refusal_flags.append(is_refusal)
            
            reward = score
            rewards.append(reward)
            
            log_entries.append({
                "sample_index": i,
                "prompt": prompt,
                "question": question,
                "completion": completion,
                "used_answer": predicted_answer,
                "best_answer": best_ans,
                "eval_method": "direct_llm_eval",
                "llm_score": score,
                "is_refusal": is_refusal,
                "judge_llm_response": response_text,
                "reward": reward,
                "timestamp": get_timestamp()
            })

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            rewards.append(0.0)
            refusal_flags.append(False)
            log_entries.append({
                "sample_index": i,
                "prompt": prompt,
                "completion": completion,
                "error": str(e),
                "is_refusal": False,
                "judge_llm_response": "ERROR",
                "reward": 0.0,
                "timestamp": get_timestamp()
            })

    save_json_output({"llm_eval_results": log_entries}, "llm_eval_reward")
    

    global _last_refusal_flags
    _last_refusal_flags = refusal_flags
    
    return rewards


_last_refusal_flags = []

def get_last_refusal_flags():
    global _last_refusal_flags
    return _last_refusal_flags
