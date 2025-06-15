import re
import logging
from typing import List
from .reward_utils import save_json_output, get_timestamp

logger = logging.getLogger(__name__)

def format_reward_func(completions: List[str], **kwargs) -> List[float]:
    rewards = []
    log_entries = []
    
    for i, completion in enumerate(completions):
        try:
            # Add <think> prefix to unify format
            if not completion.startswith("<think>"):
                completion = "<think>" + completion
            
            # Regex to check format
            regex = r"^<think>\s*(\S(?:(?!<think>|<answer>)[\s\S])*?)\s*<\/think>\s*<answer>\s*(\S(?:(?!<think>|<answer>)[\s\S])*?)\s*<\/answer>$"
            match = re.search(regex, completion, re.DOTALL)

            # Determine reward value
            if match is None or len(match.groups()) != 2:
                rewards.append(-1.0)  # Clear penalty for non-conforming format
                success = False
            else:
                rewards.append(1.0)  # Full score for conforming format
                success = True
                
            # Record detailed information
            log_entries.append({
                "sample_index": i,
                "completion": completion,
                "format_correct": success,
                "reward": rewards[-1],
                "timestamp": get_timestamp()
            })
            
        except Exception as e:
            logger.error(f"Format check error: {str(e)}")
            rewards.append(0.0)  # Return 0 on error
            log_entries.append({
                "sample_index": i,
                "completion": completion,
                "error": str(e),
                "reward": 0.0,
                "timestamp": get_timestamp()
            })
    
    # Save output to JSON file
    save_json_output({"format_checks": log_entries}, "format_reward")
            
    return rewards