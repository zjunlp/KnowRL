<div align="center">
<h1 align="center"> KnowRL </h1>
<h3 align="center"> Knowledgeable Reinforcement Learning for Factuality </h3>

<p align="center">
  <a href="https://arxiv.org/abs/25xx.xxxxx">📄arXiv</a> •
  <a href="https://huggingface.co/collections/zjunlp/knowrl-68485613feca77696d252a1d">🤗HuggingFace</a> •
  <a href="https://huggingface.co/datasets/zjunlp/KnowRL-Train-Data">📖Datasets</a>
</p>

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![](https://img.shields.io/github/last-commit/your-repo/knowrl?color=green)

</div>

## Table of Contents
- [🌻Acknowledgement](#acknowledgement)
- [🌟Overview](#overview)
- [🔧Installation](#installation)
- [📚Knowledge Base Construction](#knowledge-base-construction)
- [📉Training](#training)
- [🧐Evaluation](#evaluation)
- [🚩Citation](#citation)

---

## 🌻Acknowledgement
Our Cold-Start SFT stage is implemented based on the excellent [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) framework. We thank its authors for their great contribution!

![KnowRL Framework](https://i.imgur.com/8zD3XfX.png)

## 🌟Overview
Large Language Models (LLMs), particularly slow-thinking models, often exhibit severe hallucinations due to an inability to accurately recognize their knowledge boundaries. To address this, we propose **KnowRL**, a novel framework that integrates external knowledge into the reinforcement learning process. KnowRL guides models to perform fact-based slow thinking by incorporating a factuality reward directly into the RL training loop. This helps models learn their knowledge boundaries and fosters a more reliable, fact-based reasoning process, effectively mitigating hallucinations while maintaining or enhancing strong reasoning capabilities.

## 🔧Installation
We recommend creating a new conda environment to run our project.

```bash
conda create -n knowrl python=3.12
conda activate knowrl

git clone [https://github.com/your-repo/knowrl.git](https://github.com/your-repo/knowrl.git)
cd knowrl

pip install -r requirements.txt
```

## 📚Knowledge Base Construction
KnowRL's factuality reward relies on an external knowledge base built from a corpus like Wikipedia. This is a prerequisite for running the RL training.

1.  Place your source data file (e.g., `wikipedia.jsonl`) in a directory.
2.  Edit the `build_db.sh` script to point `DATA_PATH` to your data file.
3.  Run the script from the `build_knowledge` directory to create the SQLite database.

```bash
cd train/reward_function/FActScore/build_knowledge/
# Edit DATA_PATH in build_db.sh to your source file
bash build_db.sh
```
This will create a `knowledge_base.db` file, which is required for the `fact_reward` function during training.


## 📉Training
The training process is orchestrated by `train/train.sh`, which sets up environment variables and launches `main.py` using the configuration defined in `script/grpo.yaml`.

### 1. Configuration
Before running, you need to configure the following:

**a. Environment Variables in `train/train.sh`:**
   - Set your API keys for services like OpenAI/ZhipuAI (`OPENAI_API_KEY_FACTSCORE`, `OPENAI_API_KEY_JUDGE`).
   - Set your `WANDB_API_KEY` for experiment tracking.
   - Ensure `FACTSCORE_DB_PATH` points to the `knowledge_base.db` file you created.

**b. Training Parameters in `script/grpo.yaml`:**
   - `model_name_or_path`: Path to the base model for training (e.g., your cold-start SFT model).
   - `dataset_id_or_path`: Path to your RL training data.
   - `output_dir`: Directory to save the final trained model.
   - `wandb_project`, `wandb_entity`, `run_name`: WandB configuration.
   - `per_device_train_batch_size`, `learning_rate`, `max_steps`: Standard training hyperparameters.
   - `beta`, `num_generations`: GRPO-specific algorithm parameters.

### 2. Run Training
Once configured, launch the training from the `train` directory:

```bash
cd knowrl/train/
bash train.sh
```
The script will set the `CUDA_VISIBLE_DEVICES`, print the configuration, and start the training process.

## 🧐Evaluation
All our models are evaluated on the **OpenCompass** platform to ensure fair and reproducible results. Please refer to our paper for detailed results on benchmarks such as TruthfulQA, SimpleQA, GPQA, and AIME.

## 🚩Citation
If you find this work useful in your research, please consider citing our paper:
```bibtex
@inproceedings{ren2025knowrl,
    title={{KnowRL}: Exploring Knowledgeable Reinforcement Learning for Factuality},
    author={Baochang Ren and Shuofei Qiao and Wenhao Yu and Huajun Chen and Ningyu Zhang},
    booktitle={ACL 2025},
    year={2025}
}
```
