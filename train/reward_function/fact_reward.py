import os
import re
import sys
import logging
import traceback
from typing import List, Optional
import numpy as np

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from .FActScore.factscore.factscorer import FactScorer
from .reward_utils import save_json_output, get_timestamp

logger = logging.getLogger(__name__)

class FactualityScorer:
    def __init__(self):
        self.FACT_SCORER = None
        self._initialize_fact_scorer()
    
    def _initialize_fact_scorer(self):
        try:
            openai_api_key = os.environ.get("OPENAI_API_KEY_FACTSCORE")
            base_url = os.environ.get("OPENAI_BASE_URL_FACTSCORE")
            
            db_path = os.environ.get("FACTSCORE_DB_PATH")
            if not os.path.exists(db_path):
                logger.error(f"Knowledge base file does not exist: {db_path}")
                return None
            
            logger.info("Initializing FactScorer using standard method")
            model_name = os.environ.get("FACTSCORE_MODEL") or os.environ.get("REWARD_MODEL_NAME", "gpt-4o-mini-2024-07-18")
            fs = FactScorer(
                openai_key=openai_api_key,
                base_url=base_url,
                cache_dir=None,
                af_model_version=model_name,
                use_nli=True,
                nli_model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                nli_entailment_threshold=0.3,
                verbose=True,
            )
            fs.register_knowledge_source(db_path=db_path)
            
            self.FACT_SCORER = fs
            logger.info(f"Successfully initialized FactScorer and registered knowledge source: {db_path}")
            return self.FACT_SCORER
            
        except Exception as e:
            logger.error(f"Failed to initialize FactScorer: {str(e)}")
            return None

    def get_fact_scorer(self):
        if self.FACT_SCORER is not None:
            return self.FACT_SCORER
        
        return self._initialize_fact_scorer()

    def factuality_count_reward_func(self, prompts: List[str], completions: List[str], title: Optional[List[str]] = None, **kwargs) -> List[float]:
        rewards = [0.0] * len(completions)
        log_entries = []
        
        fs = self.get_fact_scorer()
        
        if fs is None:
            logger.error("Cannot get FactScorer instance, returning 0 for all samples")
            
            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                log_entries.append({
                    "sample_index": i,
                    "prompt": prompt,
                    "completion": completion,
                    "error": "FactScorer initialization failed",
                    "reward": 0.0,
                    "timestamp": get_timestamp()
                })
            
            save_json_output({"factuality_results": log_entries}, "factuality_reward")
            return rewards
        
        try:
            topics = []
            answers = []
            valid_indices = []
            
            for i, (prompt, completion) in enumerate(zip(prompts, completions)):
                try:
                    match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
                    
                    if match:
                        answer_text = match.group(1).strip()
                    else:
                        answer_text = completion.strip()
                    
                    if title and i < len(title) and title[i]:
                        topic = title[i]
                    else:
                        topic = prompt.strip()
                    
                    topics.append(topic)
                    answers.append(answer_text)
                    valid_indices.append(i)
                    
                except Exception as e:
                    logger.error(f"Error processing example {i}: {str(e)}")
                    log_entries.append({
                        "sample_index": i,
                        "prompt": prompt,
                        "completion": completion,
                        "error": str(e),
                        "reward": 0.0,
                        "timestamp": get_timestamp()
                    })
            
            if not valid_indices:
                logger.warning("No valid answers found")
                save_json_output({"factuality_results": log_entries}, "factuality_reward")
                return rewards
            
            # Batch evaluate factual accuracy using atomic fact count scoring mode
            fs_results = fs.get_score(
                topics=topics, 
                generations=answers,
                gamma=10,
                use_nli=True,
                use_async_af_generation=False,
                count_supported=False
            )
           
            if isinstance(fs_results, list):
                for idx, i in enumerate(valid_indices):
                    if idx < len(fs_results):
                        result = fs_results[idx]
                        
                        supported_facts_count = float(result.get("score", 0))
                        #factuality_score = min(supported_facts_count / 15.0, 1.0)
                        factuality_score = supported_facts_count
                        
                        rewards[i] = float(factuality_score)
                        
                        log_entry = {
                            "sample_index": i,
                            "topic": topics[idx],
                            "prompt": prompts[i],
                            "answer": answers[idx],
                            "has_answer_tag": re.search(r"<think>(.*?)</think>", completions[i], re.DOTALL) is not None,
                            "supported_facts_count": float(supported_facts_count),
                            "factuality_score": float(factuality_score),
                            "timestamp": get_timestamp()
                        }
                        
                        if "decisions" in result and result["decisions"] and result["decisions"][0]:
                            atomic_facts = []
                            supported_count = 0
                            total_count = 0
                            
                            for decision in result["decisions"][0]:
                                is_supported = bool(decision["is_supported"])
                                atomic_facts.append({
                                    "fact": decision["atom"],
                                    "supported": is_supported
                                })
                                total_count += 1
                                if is_supported:
                                    supported_count += 1
                            
                            log_entry["atomic_facts"] = atomic_facts
                            log_entry["supported_facts_count"] = f"{supported_count}/{total_count}"
                            log_entry["supported_facts_ratio"] = float(supported_count / total_count if total_count > 0 else 0)
                        
                        log_entries.append(log_entry)
                
                fs_summary = {
                    "overall_score": float(np.mean([r.get("score", 0) for r in fs_results])),
                    "respond_ratio": float(fs_results[0].get("respond_ratio", 1.0) if fs_results else 0),
                    "num_facts_per_response": float(np.mean([r.get("num_facts_per_response", 0) for r in fs_results]))
                }
            else:
                fs_summary = {
                    "overall_score": float(fs_results.get("score", 0)),
                    "respond_ratio": float(fs_results.get("respond_ratio", 0)),
                    "num_facts_per_response": float(fs_results.get("num_facts_per_response", 0))
                }
            
            all_results = {
                "summary": fs_summary,
                "detailed_results": log_entries
            }
            
            save_json_output(all_results, "factuality_reward")
            
        except Exception as e:
            logger.error(f"Factuality evaluation error: {str(e)}")
            error_log = {
                "error": str(e),
                "traceback": traceback.format_exc() if 'traceback' in sys.modules else None
            }
            save_json_output(error_log, "factuality_error")
            
        return rewards
