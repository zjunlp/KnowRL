import unsloth
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
# Core GRPO imports
from unsloth import FastLanguageModel

from trl import GRPOConfig, GRPOTrainer, ModelConfig, TrlParser

# Local imports
from reward_function import (
    format_reward_func,
    CombinedScorer,
    get_reward_output_dir
)
from utils import create_swanlab_callback_from_yaml, setup_logging, get_default_config_path

# Setup logging
logger = setup_logging()
# Initialize reward scorers
combined_reward = CombinedScorer()

@dataclass
class DatasetFieldMapping:
    question_field: str = "question"
    title_field: Optional[str] = "title"
    best_answer_field: Optional[str] = "answers"
    correct_answers_field: Optional[str] = None
    incorrect_answers_field: Optional[str] = None
    delimiter: str = ";"
    keep_all_fields: bool = False

@dataclass
class DatasetArguments:
    dataset_id_or_path: str = None
    dataset_splits: str = "train"
    tokenizer_name_or_path: str = None
    field_mapping: DatasetFieldMapping = field(default_factory=DatasetFieldMapping)

def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint

def process_dataset_fields(example, field_mapping):
    processed = {}
    
    question = example.get(field_mapping.question_field, "")
    processed["question"] = question
    
    if field_mapping.title_field is not None:
        title = example.get(field_mapping.title_field, "")
        processed["title"] = title
    
    if field_mapping.best_answer_field is not None:
        best_answer = example.get(field_mapping.best_answer_field, "")
        processed["best_answer"] = best_answer if best_answer else ""

    if field_mapping.keep_all_fields:
        for key, value in example.items():
            if key not in processed:
                processed[key] = value
                
    return processed

def generate_prompt(example, tokenizer):
    question = example["question"]
    
    prompt_prefix = [
        {
            "role": "user", 
            "content": f"{question}\n\nAnswer this question using the following format strictly:\n\n1. First use <think> </think> tags to show your step-by-step reasoning process.\n2. Then use <answer> </answer> tags for your final answer.\n\nExample format:\n<think>\nI need to consider these aspects...\nLooking at the evidence...\nAnalyzing step-by-step...\n</think>\n\n<answer>\nMy final, factual answer based on careful reasoning.\n</answer>\n\nImportant: Both tags must be on their own lines and your response must follow this exact format."
        },
        {
            "role": "assistant",
            "content": "I'll answer following the required format. Let me analyze this question step by step.\n\n<think>",
        },
    ]

    result = {
        "prompt": tokenizer.apply_chat_template(
            prompt_prefix,
            tokenize=False,
            continue_final_message=True
        ),
    }
    
    for key in ["best_answer", "title"]:
        if key in example:
            result[key] = example[key]
    
    return result

def grpo_function(
    model_args: ModelConfig,
    dataset_args: DatasetArguments,
    training_args: GRPOConfig,
    callbacks: List,
):
    logger.info(f"Model parameters: {model_args}")
    logger.info(f"Training parameters: {training_args}")
    logger.info(f"Dataset field mapping: {dataset_args.field_mapping}")

    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        fast_inference=True,
        load_in_4bit=False,
        max_lora_rank=model_args.lora_r,
        max_seq_length=2048,
        gpu_memory_utilization=training_args.vllm_gpu_memory_utilization,
        attn_implementation=model_args.attn_implementation,
    ) 

    # Configure PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=model_args.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=training_args.seed,
    )

    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    try:
        dataset = load_dataset('json',
            data_files=dataset_args.dataset_id_or_path, 
            split=dataset_args.dataset_splits
        )
        logger.info(f"Successfully loaded dataset with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    logger.info(f"Dataset fields: {dataset.column_names}")

    field_mapping = dataset_args.field_mapping
    
    if field_mapping.question_field not in dataset.column_names:
        raise ValueError(f"Required question field '{field_mapping.question_field}' not found in dataset")

    # Process dataset
    dataset = dataset.map(lambda x: process_dataset_fields(x, field_mapping))
    dataset = dataset.map(lambda x: generate_prompt(x, tokenizer))

    # Verify processed dataset
    required_output_fields = ["prompt"]
    for field in required_output_fields:
        if field not in dataset.column_names:
            raise ValueError(f"Required field '{field}' missing from processed dataset")

    # Print sample examples
    logger.info("Dataset sample examples:")
    for i in range(min(3, len(dataset))):
        logger.info(f"Sample {i+1}:")
        logger.info(f"Question: {dataset[i]['question']}")
        if 'title' in dataset[i]:
            logger.info(f"Title: {dataset[i]['title']}")
        logger.info(f"Prompt: {dataset[i]['prompt'][:100]}...")
        if 'best_answer' in dataset[i]:
            logger.info(f"Best Answer: {dataset[i]['best_answer']}")
        logger.info("---")

    train_dataset = dataset
    rewards_output_dir = get_reward_output_dir()

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Reward function outputs will be saved to: {os.path.abspath(rewards_output_dir)}")

    # Set up GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            combined_reward.combined_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=callbacks,
    )

    logger.info(
        f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '
        f'for {training_args.num_train_epochs} epochs ***'
    )

    # Train model
    train_result = trainer.train(
        resume_from_checkpoint=get_checkpoint(training_args) if training_args.resume_from_checkpoint else None
    )

    # Save metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training completed ***")
    logger.info("*** Saving model ***")
    
    # Save model and tokenizer
    trainer.model.config.use_cache = True
    model.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Tokenizer saved to {training_args.output_dir}")
    
    logger.info("*** Training process completed! ***")

def main():
    parser = TrlParser((ModelConfig, DatasetArguments, GRPOConfig))
    model_args, dataset_args, training_args = parser.parse_args_and_config(fail_with_unknown_args=False)
    
    config_file_path = get_default_config_path()
    swanlab_callback = create_swanlab_callback_from_yaml(config_file_path)
    
    if swanlab_callback:
        logger.info("SwanLab callback created successfully. Training metrics will be logged to SwanLab.")
        callbacks = [swanlab_callback]
    else:
        logger.info("SwanLab callback creation failed or not enabled. Training will proceed without SwanLab logging.")
        callbacks = None

    grpo_function(model_args, dataset_args, training_args, callbacks=callbacks)

if __name__ == "__main__":
    main()
