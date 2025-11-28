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
├── Trainer-Q1-DeBerta.py             # Using DeBERTa-v3 as a zero/few-shot baseline
├── Trainer-Q1-bart-large-mnli.py     # Fine-tuning BART-large-MNLI for Subtask 1 (English) 
├── Trrainer-Q1-fold5.py              # 5-fold cross-validation trainer (note: typo in filename)
│
├── data/
│   ├── train_data/train_data.json            # Official English training set
│   ├── pilot data/syllogistic_reasoning_binary_pilot_en.json
│   └── merged_data/merged_output.json        # Merged version of training + augmented data
│
├── data-augment/
│   ├── vocabulary/                           # Believable/unbelievable/gibberish terms
│   ├── schemes/                              # Syllogistic mood templates (e.g., AAA-1, EIO-3)
│   └── Q1-aug/, Q2-aug/                      # Augmented datasets for Subtask 1 & 2
│
└── data/evaluation_kit/                      # Official evaluation scripts (Subtask 1–4)


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

