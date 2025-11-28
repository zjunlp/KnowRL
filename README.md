<div align="center">
<h1 align="center"> KnowRL </h1>
<h3 align="center"> Exploring Knowledgeable Reinforcement Learning for Factuality </h3>

<p align="center">
  <a href="https://arxiv.org/abs/2506.19807">📄arXiv</a> •
  <a href="https://huggingface.co/collections/zjunlp/knowrl-68485613feca77696d252a1d">🤗HuggingFace</a> •
  <a href="https://huggingface.co/datasets/zjunlp/KnowRL-Train-Data">📖Datasets</a>
</p>

[![Awesome](https://awesome.re/badge.svg)](https://github.com/zjunlp/KnowRL)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/github/last-commit/zjunlp/KnowRL?color=green)

</div>

## Table of Contents
- [📢News](#news)
- [🌻Acknowledgement](#acknowledgement)
- [🌟Overview](#overview)
- [🔧Installation](#installation)
- [📚Knowledge Base Construction](#knowledge-base-construction)
- [📉Training](#training)
- [🚩Citation](#citation)

---
## 📢News
- **[2025-11]** 🔥 We have significantly expanded our training capabilities! In addition to the standard GRPO, we now support three advanced reinforcement learning algorithms: **DAPO [1]**, **BNPO [2]**, and **DR-GRPO [3]**.

  You can find the corresponding configuration files (`dapo.yaml`, `bnpo.yaml`, `dr_grpo.yaml`) in the `script/` directory to experiment with these new methods.

  <small>
  [1] Dapo: An open-source llm reinforcement learning system at scale <br>
  [2] BNPO: Beta Normalization Policy Optimization <br>
  [3] Understanding R1-Zero-Like Training: A Critical Perspective
  </small>
  
## 🌻Acknowledgement
Our Cold-Start SFT stage is implemented based on the excellent [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. Our reinforcement learning training code is based on [TRL](https://github.com/huggingface/trl) and [Unsloth](https://github.com/unslothai/unsloth). We thank all authors for their great contributions!
![alt text](./assets/method.jpg)

## 🌟Overview
Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucinations due to an inability to accurately recognize their knowledge boundaries. To address this, we propose **KnowRL**, a novel framework that integrates external knowledge into the reinforcement learning process. KnowRL guides models to perform fact-based slow thinking by incorporating a factuality reward directly into the RL training loop. KnowRL can be seen as leveraging a form of test-time scaling law to reduce hallucinations. This helps models learn their knowledge boundaries and fosters a more reliable, fact-based reasoning process, effectively mitigating hallucinations while maintaining or enhancing strong reasoning capabilities.

## 🔧Installation
We recommend creating a new conda environment to run our project.

```bash
conda create -n knowrl python=3.12
conda activate knowrl

git clone https://github.com/zjunlp/KnowRL.git
cd KnowRL

pip install -r requirements.txt
```

## 📚Knowledge Base Construction

KnowRL's factuality reward relies on an external knowledge base. You can either download our pre-built version or build it from your own corpus.

#### Option 1: Download Pre-built Knowledge Base (Recommended)

This is the easiest way to get started. We have hosted the pre-built `knowledge_base.db` file on [Google Drive](https://drive.google.com/uc?id=1EVFkzuFvqE8AOEcdfSSm03vvvbVDa7bI).

```bash
# The target directory for the knowledge base
cd train/reward_function/FActScore/build_knowledge/

# Download the file from Google Drive and name it knowledge_base.db
gdown https://drive.google.com/uc?id=1EVFkzuFvqE8AOEcdfSSm03vvvbVDa7bI
```
This command will download the database directly into the required folder.

#### Option 2: Build from Scratch

If you wish to build the knowledge base from your own data source (e.g., a specific Wikipedia dump).

1.  Place your source data file (e.g., `wikipedia.jsonl`) in a directory.
2.  Edit the `build_db.sh` script to point `DATA_PATH` to your data file.
3.  Run the script from the `build_knowledge` directory to create the SQLite database.

    ```bash
    cd train/reward_function/FActScore/build_knowledge/
    
    # Edit DATA_PATH in build_db.sh to point to your source file
    bash build_db.sh
    ```

This will create the `knowledge_base.db` file required for the `fact_reward` function during training.


## 📉Training
We utilize a Knowledgeable Reinforcement Learning (RL) phase to enhance the model's factuality. Also, our datasets and models have been uploaded to [huggingface](https://huggingface.co/collections/zjunlp/knowrl-68485613feca77696d252a1d).


### Knowledgeable Reinforcement Learning 
This stage uses the SFT-tuned model and further trains it with our knowledge-enhanced reward signal. The process is orchestrated by `train/train.sh`, which launches `main.py` using the configuration defined in `script/grpo.yaml`. We are training two 7B models, `DeepSeek-R1-Distill-Qwen-7B` and `Skywork-OR1-7B-Preview`, on 1×A800 GPU.

**a. Environment Variables in `train/train.sh`:**
This script sets up all necessary environment variables and executes the training.
   - Set your API keys for services like OpenAI (`OPENAI_API_KEY_FACTSCORE`, `OPENAI_API_KEY_JUDGE`).
   - Set your `SWANLAB_API_KEY` for experiment tracking.
   - Ensure `FACTSCORE_DB_PATH` points to the `knowledge_base.db` file you created.

**b. Training Parameters in `script/grpo.yaml`**
This file contains all hyperparameters for the RL stage.
   - `model_name_or_path`: Path to the base model for RL training (this should be your SFT-tuned model).
   - `dataset_id_or_path`: Path to your RL training data.
   - `output_dir`: Directory to save the final trained model.
   - `swanlab_project`, `swanlab_experiment_name`: SwanLab configuration.
   - `per_device_train_batch_size`, `learning_rate`, `max_steps`: Standard training hyperparameters.
   - `beta`, `num_generations`: GRPO-specific algorithm parameters.

**c. Launch RL Training**
Once configured, launch the training from the `train` directory:

```bash
cd KnowRL/train/
bash train.sh
```
The script will set the `CUDA_VISIBLE_DEVICES`, print the configuration, and start the training process.

<details>
<summary>Click to view train.sh</summary>

```bash
#!/bin/bash
# ============================================================================
# API Configuration - Replace with your actual credentials
# ============================================================================
export OPENAI_API_KEY_FACTSCORE="your_openai_api_key_here"
export OPENAI_BASE_URL_FACTSCORE="https://api.openai.com/v1"

export OPENAI_API_KEY_JUDGE="your_openai_api_key_here"
export OPENAI_API_BASE_JUDGE="https://api.openai.com/v1"

export SWANLAB_API_KEY="your_swanlab_api_key_here"
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
```
</details>



## 🚩Citation
If you find this work useful in your research, please consider citing our paper:
```bibtex
@article{ren2025knowrl,
  title={KnowRL: Exploring Knowledgeable Reinforcement Learning for Factuality},
  author={Ren, Baochang and Qiao, Shuofei and Yu, Wenhao and Chen, Huajun and Zhang, Ningyu},
  journal={arXiv preprint arXiv:2506.19807},
  year={2025}
}
```
