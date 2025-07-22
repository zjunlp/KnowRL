import os
import json
import logging
import yaml
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import os
logger = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
GRPO_path = os.path.join(current_dir, '..', 'script', 'grpo.yaml')

def load_yaml_config(yaml_path=GRPO_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {yaml_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {yaml_path}: {str(e)}, cannot continue")
        sys.exit(1)


_timestamp_singleton = None

def get_output_timestamp():
    global _timestamp_singleton
    if _timestamp_singleton is None:
        _timestamp_singleton = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _timestamp_singleton

def get_reward_output_dir():
    yaml_config = load_yaml_config()
    project_name = yaml_config.get("swanlab_project", "default_project")
    
    timestamp = get_output_timestamp()
    output_dir = os.path.join("reward_outputs", project_name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created reward function output directory: {output_dir}")
    return output_dir

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_json_output(data: Dict, function_name: str, sample_index: int = None):
    rewards_output_dir = get_reward_output_dir()
    
    if sample_index is not None:
        filename = f"{function_name}_sample_{sample_index}.json"
    else:
        filename = f"{function_name}.json"
    
    filepath = os.path.join(rewards_output_dir, filename)
    
    existing_data = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Cannot parse existing JSON file: {filepath}, will create new file")
    
    # Merge data
    if isinstance(data, dict) and isinstance(existing_data, dict):
        for key, value in data.items():
            if key in existing_data and isinstance(value, list) and isinstance(existing_data[key], list):
                existing_data[key].extend(value)
            else:
                existing_data[key] = value
        merged_data = existing_data
    else:
        merged_data = data
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    return filepath
