import os
import yaml
import wandb
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def initialize_wandb_from_yaml(yaml_file_path: str):
    """Initialize WandB from YAML configuration file"""
    try:
        if not os.path.isabs(yaml_file_path):
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            full_yaml_path = os.path.join(current_script_dir, yaml_file_path)
        else:
            full_yaml_path = yaml_file_path

        if not os.path.exists(full_yaml_path):
            logger.error(f"YAML config file '{full_yaml_path}' not found.")
            return None

        with open(full_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        report_to = config.get('report_to')
        if report_to != 'wandb':
            logger.info(f"WandB reporting not enabled in YAML (report_to: {report_to}). Skipping WandB initialization.")
            return None

        wandb_project = config.get('wandb_project')
        wandb_entity = config.get('wandb_entity')
        run_name = config.get('run_name')

        if not all([wandb_project, wandb_entity, run_name]):
            logger.error("Missing WandB parameters in YAML file.")
            logger.error(f"  - wandb_project: {wandb_project}")
            logger.error(f"  - wandb_entity: {wandb_entity}")
            logger.error(f"  - run_name: {run_name}")
            return None

        
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config  
        )
        
        logger.info(f"WandB successfully initialized!")
        logger.info(f"  Project: {wandb_project}")
        logger.info(f"  Entity: {wandb_entity}")
        logger.info(f"  Run Name: {run_name}")
        logger.info(f"  URL: {wandb.run.get_url()}")
        
        return wandb

    except FileNotFoundError:
        logger.error(f"YAML config file '{full_yaml_path}' not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file '{full_yaml_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unknown error initializing WandB: {e}")
        return None


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger

def load_yaml_config(yaml_path: str):
    try:
        # 支持相对路径和绝对路径
        if not os.path.isabs(yaml_path):
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            full_yaml_path = os.path.join(current_script_dir, yaml_path)
        else:
            full_yaml_path = yaml_path
            
        with open(full_yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {full_yaml_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config file {yaml_path}: {str(e)}")
        raise

def get_default_config_path():
    return "./script/grpo.yaml"