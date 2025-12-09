# semeval-2026-task11
Official implementation for SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Language Models
# SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Language Models

> **Team**: YNU-HPCC 
> **Task**: [SemEval-2026 Task 11](https://sites.google.com/view/semeval-2026-task11)  
> **Goal**: Improve model robustness to *content effects* in syllogistic reasoning by decoupling world knowledge from logical form.

This repository contains our official implementation for **SemEval-2026 Task 11**. We focus on **data augmentation with synthetic, content-neutral syllogisms** to train models that rely on formal logic rather than semantic plausibility.

##  Approach Overview

We hypothesize that exposing models to **implausible but valid** or **plausible but invalid** syllogisms during training reduces reliance on world knowledge. To this end, we generate synthetic data using:

- **Gibberish vocabulary**: Replace real terms with meaningless tokens (e.g., "All A are B").
- **Formal schemes**: Enumerate all 64 syllogistic moods across 4 figures.
- **Plausibility-Validity combinations**: Systematically create examples covering:
  - Valid + Plausible
  - Valid + Implausible
  - Invalid + Plausible
  - Invalid + Implausible

This forces the model to learn the underlying logical structure, not surface-level semantics.

## 📁 Repository Structure

- `Trainer-Q1-DeBerta.py` — 使用 DeBERTa-v3 作为零样本/少样本基线
- `Trainer-Q1-bart-large-mnli.py` — 针对子任务 1 微调 BART-large-MNLI 模型
- `Trainer-Q1-fold5.py` — 5 折交叉验证训练代码
- `data/train_data/train_data.json` — 官方英语训练集
- `data/pilot data/syllogistic_reasoning_binary_pilot_en.json` — 官方测试数据
- `data/merged_data/merged_output.json` — 合并后的训练 + 增强数据
- `data-augment/vocabulary/` — 可信/不可信/胡言乱语术语
- `data-augment/schemes/` — 三段论语气模板（如 AAA-1, EIO-3）
- `data-augment/Q1-aug/, Q2-aug/` — 子任务 1 和 2 的增强数据集
- `data/evaluation_kit/` — 官方评估脚本（子任务 1–4）


##  How to Run

### 1. Environment Setup
```bash
pip install torch transformers datasets scikit-learn

python  Trainer-Q1-bart-large-mnli.py   \
  --train_file dat-augment/Q1-aug/unvalidity_plausibility.json/ dat-augment/Q1-aug/unvalidity_unplausibility.json/ dat-augment/Q1-aug/validity_plausibility.json/
            dat-augment/Q1-aug/validity_unplausibility.json/
  --test_file data/merged_data/merged_output.json \
  --model_name microsoft/-bart-large-mnli \
  --output_dir ./results/-bart-large-mnli_q1_aug \

## 🧠 新增模块：`syllogism_rlvr` —— 基于强化学习的形式推理模型

我们新增了 `syllogism_rlvr` 模块，用于实现通过强化学习（RL）提升模型对三段论形式有效性的判断能力。该模块专注于解决语义合理性（plausibility）与逻辑有效性（validity）之间的混淆问题。
syllogism_rlvr/
├── sft_syllogism_full/      # 存放 SFT 训练完成后的模型权重文件
├── Asda.py                  # 辅助脚本
├── eval_syllogism.py        # 评估标准监督微调（SFT）模型的性能
├── evaluate_rlr_model.py    # 评估经过 RL 训练的 RLR 模型效果
├── ppo_data.py              # 构造 PPO 强化学习所需的数据格式（包括 reward 设计）
├── rlvr.py                  # 定义 RLVR策略网络、reward 函数和训练流程
└── sft.py                   # 执行监督微调（Supervised Fine-Tuning）的主脚本
目前经过实验的模型准确率
microsoft/deberta-v2-xlarge-mnli + lora 78.37%
bart-large-mnli 92.30%
bart-large-mnli + 五折交叉   92.11%
mistralai/Mistral-7B-Instruct-v0.3 84%
