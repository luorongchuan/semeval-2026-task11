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

- `Q1` — Experiment code for Task 1
- `Q3` — Experiment code for Task 3
- `data/train_data/train_data.json` — Official English training set
- `data/pilot data/syllogistic_reasoning_binary_pilot_en.json` — Official test data
- `data-augment/vocabulary/` — Trustworthy/Untrustworthy Vocabulary
- `data-augment/schemes/` — Syllogism Mood Templates (e.g., AAA-1, EIO-3)
- `data-augment/Q1-aug/, Q2-aug/` — Augmented Datasets for Subtasks 1 and 2
- `data/evaluation_kit/` — Official Evaluation Script (Subtasks 1–4)
