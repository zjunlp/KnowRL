# ============================================================================
# API Configuration - Replace with your actual credentials
# ============================================================================
export OPENAI_API_KEY_FACTSCORE="your_openai_api_key_here"
export OPENAI_BASE_URL_FACTSCORE="https://api.openai.com/v1"

export OPENAI_API_KEY_JUDGE="your_openai_api_key_here"
export OPENAI_API_BASE_JUDGE="https://api.openai.com/v1"

export WANDB_API_KEY="your_wandb_api_key_here"
export WANDB_MODE="offline" ## Optional
# ============================================================================
# Configuration
# ============================================================================
export FACTSCORE_DB_PATH="./FActScore/build_knowledge/knowledge_base.db"
export USE_API_MANAGER_FOR_LLM_EVAL=True
export USE_API_MANAGER_FOR_FACTSCORE=True

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

# Configuration file
CONFIG_FILE="./script/grpo.yaml"

# ============================================================================
# Run Training
# ============================================================================
echo "Starting GRPO training..."
echo "Config: $CONFIG_FILE"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python main.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed!"
    exit 1
fi