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
        
        # Define refusal patterns
        self.refusal_patterns = [
            r"(?i)\b(I cannot|I can't|I am unable to)\s+(answer|respond|provide|give|tell|assist|help)\b",
            r"(?i)\b(cannot|can't|unable to)\s+(answer|respond|provide|give|tell|assist|help)\b",
            r"(?i)\b(I must|I have to|I will have to|I need to)\b(.*?)\b(decline|refuse|refrain from|abstain from)\b\s+(answering|responding|providing|giving)\b",
            r"(?i)\b(as an AI|as a language model|as an assistant),?\s+I\s+(cannot|can't|am unable to|must decline|am not able to|don't have|won't)\b",
            r"(?i)\b(ethical|moral|legal)\b(.*?)\b(concerns|issues|considerations|limitations|constraints)\b(.*?)\b(prevent|stop|preclude|prohibit|disallow)\b",
            r"(?i)\b(would need more|need more|require more|need additional|require additional)\s+(information|details|context|specifics|clarification)\s+to\s+(answer|respond|provide|help|assist)\b",
            r"(?i)\b(insufficient information|lack of information|lack of context|lack of details|not enough information|not enough context|too vague)\s+to\s+(answer|respond|provide|give|help|assist)\b",
            r"(?i)\b(more information|additional information|more context|additional context|more details|additional details)\s+would be\s+(needed|required|necessary)\b"
        ]
        
        # Define common stop words for keyword extraction
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
            'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 
            'from', 'of', 'that', 'this', 'these', 'those', 'it', 'they', 
            'he', 'she', 'we', 'you', 'i', 'me', 'my', 'your', 'our', 'their'
        }
        
    def is_refusal_response(self, text: str) -> bool:
        if not text or text.isspace():
            return True
        
        for pattern in self.refusal_patterns:
            if re.search(pattern, text):
                return True
        
        return False
        
    def contains_numbers(self, text: str) -> bool:
        return bool(re.search(r'\d', text))
        
    def has_partial_match(self, answer: str, best_answer: str) -> bool:
        if not best_answer or not answer:
            return False
        
        best_answer_words = set()
        for word in re.findall(r'\w+', best_answer):
            if word.lower() not in self.stop_words and len(word) > 1:
                best_answer_words.add(word.lower())
        
        answer_lower = answer.lower()
        for word in best_answer_words:
            if word in answer_lower:
                return True
        
        return False
    
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
                    score = -1.0
                else:
                    if re.search(correct_pattern, response_text):
                        score = 2.0
                    elif re.search(incorrect_pattern, response_text):
                        score = -1.0
                    elif re.search(not_attempted_pattern, response_text):
                        score = -1.0
                    else:
                        logger.warning(f"Unrecognized scoring result: '{raw_response_text}'")
                        score = 0.0
            else:
                if re.search(correct_pattern, response_text):
                    score = 2.0
                elif re.search(incorrect_pattern, response_text):
                    score = -1.0
                elif re.search(not_attempted_pattern, response_text):
                    score = -1.0
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
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "completion": completion,
                    "error": "answer tag not found",
                    "reward": -1,
                    "judge_llm_response": "N/A",
                    "timestamp": get_timestamp()
                })
                continue
            predicted_answer = match.group(1).strip()

            if not predicted_answer or predicted_answer.isspace():
                rewards.append(-1)
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "completion": completion,
                    "error": "empty content in answer tag",
                    "reward": -1,
                    "raw_llm_response": "N/A",
                    "timestamp": get_timestamp()
                })
                continue

            best_ans = best_answer[i] if best_answer and i < len(best_answer) else None
            question = qwestion[i] if qwestion and i < len(qwestion) else prompt

            # Step 1: Refusal detection
            if judge_llm.is_refusal_response(predicted_answer):
                reward = -1.0
                rewards.append(reward)
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "question": question,
                    "completion": completion,
                    "used_answer": predicted_answer,
                    "best_answer": best_ans,
                    "eval_method": "refusal_detection",
                    "reward": reward,
                    "timestamp": get_timestamp()
                })
                continue
            
            # Step 2: Check if best answer contains numbers
            if best_ans and judge_llm.contains_numbers(best_ans):
                score, response_text = judge_llm.get_score(
                    question,
                    predicted_answer,
                    best_ans,
                    prompt_template=LLM_EVAL_PROMPT
                )
                
                reward = score
                rewards.append(reward)
                
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "question": question,
                    "completion": completion,
                    "used_answer": predicted_answer,
                    "best_answer": best_ans,
                    "eval_method": "llm_eval_numeric",
                    "llm_score": score,
                    "judge_llm_response": response_text,
                    "reward": reward,
                    "timestamp": get_timestamp()
                })
                continue
            
            # Step 3: Partial match detection
            if best_ans and not judge_llm.has_partial_match(predicted_answer, best_ans):
                reward = -1.0
                rewards.append(reward)
                
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "question": question,
                    "completion": completion,
                    "used_answer": predicted_answer,
                    "best_answer": best_ans,
                    "eval_method": "keyword_mismatch",
                    "reward": reward,
                    "timestamp": get_timestamp()
                })
                continue
            
            # Step 4: Only use LLM evaluation when there's a partial match
            score, response_text = judge_llm.get_score(
                question,
                predicted_answer,
                best_ans,
                prompt_template=LLM_EVAL_PROMPT
            )
            
            reward = score
            rewards.append(reward)
            
            log_entries.append({
                "sample_index": i,
                "prompt": prompt,
                "question": question,
                "completion": completion,
                "used_answer": predicted_answer,
                "best_answer": best_ans,
                "eval_method": "llm_eval_after_partial_match",
                "llm_score": score,
                "judge_llm_response": response_text,
                "reward": reward,
                "timestamp": get_timestamp()
            })

        except Exception as e:
            logger.error(f"Evaluation error: {str(e)}")
            rewards.append(0.0)
            log_entries.append({
                "sample_index": i,
                "prompt": prompt,
                "completion": completion,
                "error": str(e),
                "judge_llm_response": "ERROR",
                "reward": 0.0,
                "timestamp": get_timestamp()
            })

    save_json_output({"llm_eval_results": log_entries}, "llm_eval_reward")
    return rewards