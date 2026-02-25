# ========== 放在文件最顶部：外网镜像 & 可选 Token ==========

import os
# 不再需要 HF_ENDPOINT，因为我们完全离线
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# ========== 正式导入 ==========

import logging
import re
import math
import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# -----------------------------
# 评估指标
# -----------------------------

def compute_metrics(p):
    logits = p.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    if isinstance(logits, list):
        logits = np.concatenate([np.asarray(x) for x in logits], axis=0)

    preds = np.argmax(logits, axis=-1)
    return {"eval_accuracy": accuracy_score(p.label_ids, preds)}

# -----------------------------
# 配置与数据预处理
# -----------------------------

_SENT_SPLIT = re.compile(r'\s*(?<=[\.\?!。；;])\s+')

@dataclass
class NLIConfig:
    local_model_path: str = "/home/luorongchuan/workspace_134/Semeval2026/Q3/local-model/mdeberta"
    max_length: int = 512
    entail_thresh: float = 0.5
    contra_guard: float = 0.4
    device: Optional[str] = None  # "cuda" / "cpu" / None 自动
    show_confusion: bool = True
    batch_size: int = 32
    epochs: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42

# -----------------------------
# 数据集处理
# -----------------------------

def safe_split_syllogism(s: str) -> Tuple[str, str, str]:
    text = s.strip()
    if not text:
        return "", "", ""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]

    def _first_two_delims(t: str) -> Tuple[int, int]:
        cand = []
        for ch in ['.', '。']:
            i = t.find(ch)
            if i != -1:
                cand.append(i)
        i = min(cand) if cand else -1
        j = -1
        if i != -1:
            rest = t[i+1:]
            cand2 = []
            for ch in ['.', '。']:
                ii = rest.find(ch)
                if ii != -1:
                    cand2.append(ii)
            j = (i+1 + min(cand2)) if cand2 else -1
        return i, j

    i, j = _first_two_delims(text)
    p1 = text[:i].strip() if i != -1 else text
    p2 = text[i+1:j].strip() if j != -1 else ""
    c  = text[j+1:].strip() if j != -1 else ""
    return p1, p2, c

# -----------------------------
# 数据加载
# -----------------------------

def _to_bin_bool(x):
    if x is None:
        return None
    if isinstance(x, str):
        xs = x.strip().lower()
        if xs in {"plausible", "valid", "true", "yes", "1"}:
            return 1
        if xs in {"implausible", "invalid", "false", "no", "0"}:
            return 0
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, (int, np.integer)):
        return 1 if int(x) == 1 else 0
    return None

def load_and_process_data(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)

    processed_data = []
    for example in data:
        premise1, premise2, conclusion = safe_split_syllogism(example.get('syllogism', ''))

        label = _to_bin_bool(example.get('validity', 0))
        plaus = _to_bin_bool(example.get("plausibility"))

        processed_data.append({
            'premise1': premise1 or "",
            'premise2': premise2 or "",
            'conclusion': conclusion or "",
            'labels': label,
            'plausibility': plaus
        })
    return processed_data

# -----------------------------
# 内容效应计算
# -----------------------------

def compute_content_effects(preds, labels, plaus):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    plaus = np.asarray(plaus)

    def _safe_acc(preds, labels, mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.nan
        return np.mean((preds[idx] == labels[idx]).astype(np.float32))

    # IPCE
    diffs_ip = []
    for p in [0, 1]:
        m_p = (plaus == p)
        if m_p.sum() == 0:
            continue
        acc_valid = _safe_acc(preds, labels, m_p & (labels == 1))
        acc_invalid = _safe_acc(preds, labels, m_p & (labels == 0))
        if not np.isnan(acc_valid) and not np.isnan(acc_invalid):
            diffs_ip.append(abs(acc_valid - acc_invalid))
    ipce = np.mean(diffs_ip) if diffs_ip else np.nan

    # CPCE
    diffs_cp = []
    for fv_val in [0, 1]:
        m_fv = (labels == fv_val)
        if m_fv.sum() == 0:
            continue
        acc_plaus = _safe_acc(preds, labels, m_fv & (plaus == 1))
        acc_implaus = _safe_acc(preds, labels, m_fv & (plaus == 0))
        if not np.isnan(acc_plaus) and not np.isnan(acc_implaus):
            diffs_cp.append(abs(acc_plaus - acc_implaus))
    cpce = np.mean(diffs_cp) if diffs_cp else np.nan

    overall_acc = np.mean((preds == labels).astype(np.float32))
    tce = (ipce + cpce) / 2.0 if not np.isnan(ipce) and not np.isnan(cpce) else np.nan

    log_penalty = math.log(1 + tce) if tce >= 0 else 0
    acc_over_tce = overall_acc / (1 + log_penalty)

    return {
        "ipce": ipce,
        "cpce": cpce,
        "tce": tce,
        "accuracy": overall_acc,
        "acc_over_tce": acc_over_tce
    }

def create_stratify_labels(labels, plausibilities):
    stratify = []
    for y, p in zip(labels, plausibilities):
        if p is None or pd.isna(p):
            p = -1
        stratify.append(f"{y}_{p}")
    return stratify

# -----------------------------
# 微调模型
# -----------------------------

def train_model(data, cfg: NLIConfig):
    labels = [d['labels'] for d in data]
    plaus = [d['plausibility'] for d in data]
    stratify_labels = create_stratify_labels(labels, plaus)

    from collections import Counter
    counts = Counter(stratify_labels)
    min_count = min(counts.values())
    if min_count < 2:
        print("⚠️ 警告：某些 (label, plaus) 组合样本数 < 2，无法严格分层。")
        stratify_for_split = labels
    else:
        stratify_for_split = stratify_labels

    train_data, val_data = train_test_split(
        data,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=stratify_for_split
    )

    # 加载本地 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.local_model_path, use_fast=False)

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

    def preprocess_function(examples):
        instr = "Evaluate based on formal logic only. Ignore real-world plausibility."
        premises = [f"{instr}\n\nPremise 1: {p1}\nPremise 2: {p2}" for p1, p2 in zip(examples["premise1"], examples["premise2"])]
        hypotheses = [f"Conclusion: {c}" for c in examples["conclusion"]]
        enc = tokenizer(premises, hypotheses, padding=True, truncation=True, max_length=cfg.max_length)
        enc["labels"] = examples["labels"]
        return enc

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["premise1", "premise2", "conclusion"])
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["premise1", "premise2", "conclusion"])

    print("Train dataset shape:", train_dataset.shape)
    print("Val dataset shape:", val_dataset.shape)

    # 加载本地模型
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.local_model_path,
        num_labels=2,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
        use_safetensors=True
    ).to(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model.config.label2id = {"invalid": 0, "valid": 1}
    model.config.id2label = {0: "invalid", 1: "valid"}

    print("Model config num_labels:", model.config.num_labels)
    print("Classifier:", model.classifier)

    training_args = TrainingArguments(
        output_dir='/home/luorongchuan/workspace_134/Semeval2026/Q3/local-code/mdeberta/out',
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=cfg.batch_size,
        warmup_steps=0,
        weight_decay=cfg.weight_decay,
        report_to=["none"],
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("/home/luorongchuan/workspace_134/Semeval2026/Q3/local-code/mdeberta/out")
    tokenizer.save_pretrained("/home/luorongchuan/workspace_134/Semeval2026/Q3/local-code/mdeberta/out")

    eval_results = trainer.evaluate()

    pred_out = trainer.predict(val_dataset)
    pred_logits = pred_out.predictions
    if isinstance(pred_logits, tuple):
        pred_logits = pred_logits[0]
    y_pred = np.argmax(pred_logits, axis=-1)
    y_true = pd.DataFrame(val_data)["labels"].to_numpy(dtype=int)
    plaus_arr = pd.DataFrame(val_data)["plausibility"].to_numpy(dtype=int)

    content_metrics = compute_content_effects(
        preds=y_pred,
        labels=y_true,
        plaus=plaus_arr
    )

    print(
        "[Content Bias Metrics] "
        f"IPCE={content_metrics['ipce']:.6f}, "
        f"CPCE={content_metrics['cpce']:.6f}, "
        f"TCE={content_metrics['tce']:.6f}, "
        f"ACC/TCE={content_metrics['acc_over_tce']:.6f}"
    )

    eval_results.update({
        "ipce": content_metrics["ipce"],
        "cpce": content_metrics["cpce"],
        "tce": content_metrics["tce"],
        "acc_over_tce": content_metrics["acc_over_tce"],
    })

    return eval_results

# -----------------------------
# 测试集预测（完全从本地加载）
# -----------------------------

def predict_on_test_set(test_filepath: str, model_path: str, output_json_path: str, max_length: int = 512):
    """
    从 model_path 同时加载 tokenizer 和 model（要求 tokenizer 已保存到该目录）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    with open(test_filepath, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions = []

    for item in tqdm(test_data, desc="Predicting on test set"):
        syllogism = item.get("syllogism", "").strip()
        _id = item.get("id")

        if not syllogism or _id is None:
            print(f"⚠️ 跳过无效条目: {item}")
            continue

        premise1, premise2, conclusion = safe_split_syllogism(syllogism)

        instr = "Based on formal logic only, does the conclusion follow? Ignore real-world truth."
        premise_text = f"{instr}\n\nPremise 1: {premise1}\nPremise 2: {premise2}"
        hypothesis_text = f"Conclusion: {conclusion}"

        inputs = tokenizer(
            premise_text,
            hypothesis_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_label = torch.argmax(logits, dim=-1).item()

        predictions.append({
            "id": _id,
            "validity": bool(pred_label)
        })

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ 预测完成，结果已保存至: {output_json_path}")

# -----------------------------
# 主程序
# -----------------------------

if __name__ == "__main__":
    train_data_path = '/home/luorongchuan/workspace_134/Semeval2026/Q3/data/q3_merge_translate.json'
    data = load_and_process_data(train_data_path)

    cfg = NLIConfig(
        local_model_path="/home/luorongchuan/workspace_134/Semeval2026/Q3/local-model/mdeberta",
        max_length=512,
        entail_thresh=0.5,
        contra_guard=0.4,
        device="cuda",
        batch_size=32,
        epochs=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    eval_results = train_model(data, cfg)
    print("Evaluation results:", eval_results)

    # 预测阶段
    TEST_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q3/data/test_data_subtask_3.json"
    OUTPUT_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q3/local-code/mdeberta/predictions.json"
    MODEL_SAVE_DIR = "/home/luorongchuan/workspace_134/Semeval2026/Q3/local-code/mdeberta/out"

    if os.path.exists(TEST_FILE):
        predict_on_test_set(
            test_filepath=TEST_FILE,
            model_path=MODEL_SAVE_DIR,
            output_json_path=OUTPUT_FILE,
            max_length=cfg.max_length
        )
    else:
        print(f"⚠️ 测试文件不存在: {TEST_FILE}，跳过预测。")