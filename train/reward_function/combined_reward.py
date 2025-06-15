import logging
from typing import List, Optional
from .correct_reward import llm_eval_reward_func
from .fact_reward import FactualityScorer
from .reward_utils import save_json_output, get_timestamp

logger = logging.getLogger(__name__)

class CombinedScorer:
    def __init__(self):
        self.factuality_scorer = FactualityScorer()
    
    def combined_reward_func(self, prompts: List[str], completions: List[str], best_answer: Optional[List[str]] = None, 
                           qwestion: Optional[List[str]] = None, title: Optional[List[str]] = None, **kwargs) -> List[float]:
        # Perform LLM scoring - directly call global function
        llm_rewards = llm_eval_reward_func(prompts, completions, best_answer, qwestion, **kwargs)
        
        # Perform factuality scoring
        factuality_rewards = self._factuality_eval(prompts, completions, title)
        
        # Merge results - no need to pass logs, directly use reward scores
        return self._merge_results(prompts, completions, llm_rewards, factuality_rewards)
    
    def _merge_results(self, prompts: List[str], completions: List[str], llm_rewards: List[float], factuality_rewards: List[float]) -> List[float]:
        combined_log_entries = []
        final_rewards = llm_rewards.copy()
        
        for i in range(len(completions)):
            fact_score = factuality_rewards[i] if i < len(factuality_rewards) else 0.0
            llm_reward = llm_rewards[i] if i < len(llm_rewards) else 0.0
            
            # If LLM score is -1, adjust reward value
            if llm_reward == -1:
                final_rewards[i] = -1 + fact_score
            
            log_entry = {
                "sample_index": i,
                "prompt": prompts[i] if i < len(prompts) else "",
                "completion": completions[i] if i < len(completions) else "",
                "llm_reward": llm_reward,
                "factuality_score": fact_score,
                "combined_reward": final_rewards[i],
                "timestamp": get_timestamp()
            }
            
            combined_log_entries.append(log_entry)
        
        save_json_output({"combined_eval_results": combined_log_entries}, "combined_reward")
        
        return final_rewards
    
    def _factuality_eval(self, prompts: List[str], completions: List[str], title: Optional[List[str]] = None) -> List[float]:
        return self.factuality_scorer.factuality_count_reward_func(prompts, completions, title)