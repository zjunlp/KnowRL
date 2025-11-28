#!/bin/bash

# ============================================================================
# API Configuration - Please replace with your actual credentials
# ============================================================================
export OPENAI_API_KEY_FACTSCORE="your_factscore_api_key_here"
export OPENAI_BASE_URL_FACTSCORE="https://your.api.endpoint/v1/"

export OPENAI_API_KEY_JUDGE="your_judge_api_key_here"
export OPENAI_API_BASE_JUDGE="https://your.api.endpoint/v1/"

export SWANLAB_API_KEY="your_swanlab_api_key_here"

# ============================================================================
# Global Configuration
# ============================================================================
export FACTSCORE_DB_PATH="./FActScore/build_knowledge/knowledge_base.db"
export USE_API_MANAGER_FOR_LLM_EVAL=True
export USE_API_MANAGER_FOR_FACTSCORE=True

# Set GPU device
export CUDA_VISIBLE_DEVICES=1

# ============================================================================
# Training Configuration Selection
# ============================================================================
# Please select the configuration file name here.
# Available options:
#   - "grpo.yaml"     : Standard GRPO configuration
#   - "dapo.yaml"     : Standard DAPO configuration
#   - "bnpo.yaml"     : BNPO configuration (Batch-Normalized)
#   - "dr_grpo.yaml"  : DR-GRPO configuration (Distribution-Robust)
CONFIG_NAME="grpo.yaml"

# Construct the full path and export variables
CONFIG_FILE="./script/${CONFIG_NAME}"
export TRAIN_CONFIG_FILE="${CONFIG_NAME}"

# ============================================================================
# Run Training
# ============================================================================
echo "Starting training process..."
echo "Selected Config: $CONFIG_NAME"
echo "Full Config Path: $CONFIG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python main.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed!"
    exit 1
fi
