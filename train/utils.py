import os
import yaml
import logging
from typing import Optional
from swanlab.integration.transformers import SwanLabCallback

logger = logging.getLogger(__name__)

def create_swanlab_callback_from_yaml(yaml_file_path: str):
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
        if report_to != 'swanlab':
            logger.info(f"SwanLab reporting not enabled in YAML (report_to: {report_to}). Skipping SwanLab initialization.")
            return None

        swanlab_project = config.get('swanlab_project')
        swanlab_experiment_name = config.get('swanlab_experiment_name')
        swanlab_workspace = config.get('swanlab_workspace', None)  

        if not all([swanlab_project, swanlab_experiment_name]):
            logger.error("Missing SwanLab parameters in YAML file.")
            logger.error(f"  - swanlab_project: {swanlab_project}")
            logger.error(f"  - swanlab_experiment_name: {swanlab_experiment_name}")
            return None

        callback_kwargs = {
            'project': swanlab_project,
            'experiment_name': swanlab_experiment_name,
        }
        
        if swanlab_workspace:
            callback_kwargs['workspace'] = swanlab_workspace

        swanlab_callback = SwanLabCallback(**callback_kwargs)
        
        logger.info(f"SwanLab callback successfully created!")
        logger.info(f"  Project: {swanlab_project}")
        logger.info(f"  Experiment Name: {swanlab_experiment_name}")
        if swanlab_workspace:
            logger.info(f"  Workspace: {swanlab_workspace}")
        
        return swanlab_callback

    except FileNotFoundError:
        logger.error(f"YAML config file '{full_yaml_path}' not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML file '{full_yaml_path}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unknown error creating SwanLab callback: {e}")
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
