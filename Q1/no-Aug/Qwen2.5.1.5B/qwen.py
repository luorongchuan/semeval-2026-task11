# ========== 放在文件最顶部：外网镜像 & 可选 Token ==========

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_CACHE"] = "/tmp/hf_cache"  # 使用临时目录避免权限问题
os.environ["TRANSFORMERS_OFFLINE"] = "false"
os.environ["HF_HUB_OFFLINE"] = "false"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# ========== 正式导入 ==========

import logging
import re
import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# -----------------------------
# 自定义 Qwen 分类模型（核心改动）
# -----------------------------

from transformers import Qwen2Model

class QwenForSequenceClassification(nn.Module):
    def __init__(self, model_name_or_path: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_labels = num_labels
        self.qwen = Qwen2Model.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.qwen.config.hidden_size, num_labels)
        # Ensure pad token is set in config
        if self.qwen.config.pad_token_id is None:
            self.qwen.config.pad_token_id = self.qwen.config.eos_token_id

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        # Get last non-padding token representation
        sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch]
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        last_hidden = outputs.last_hidden_state[batch_indices, sequence_lengths]

        pooled_output = self.dropout(last_hidden)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


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
    model_name: str = "Qwen/Qwen2.5-1.5B"  # ← 改为 Qwen
    max_length: int = 512
    entail_thresh: float = 0.5
    contra_guard: float = 0.4
    device: Optional[str] = None
    show_confusion: bool = True
    batch_size: int = 8          # ← 减小 batch size
    epochs: int = 5
    learning_rate: float = 1e-5  # ← 更小学习率
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
        if not (np.isnan(acc_valid) or np.isnan(acc_invalid)):
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
        if not (np.isnan(acc_plaus) or np.isnan(acc_implaus)):
            diffs_cp.append(abs(acc_plaus - acc_implaus))
    cpce = np.mean(diffs_cp) if diffs_cp else np.nan

    overall_acc = np.mean((preds == labels).astype(np.float32))
    tce = (ipce + cpce) / 2.0 if not (np.isnan(ipce) or np.isnan(cpce)) else np.nan

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
# 微调模型（主训练函数）
# -----------------------------

def train_model(data, cfg: NLIConfig):
    labels = [d['labels'] for d in data]
    plaus = [d['plausibility'] for d in data]
    stratify_labels = create_stratify_labels(labels, plaus)
    from collections import Counter
    counts = Counter(stratify_labels)
    min_count = min(counts.values()) if counts else 0
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

    # 初始化 tokenizer（Qwen 需要 trust_remote_code）
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        # 构造单文本输入（Qwen 是 decoder，不依赖 [SEP]）
        texts = [
            f"Premise 1: {p1}\nPremise 2: {p2}\nConclusion: {c}"
            for p1, p2, c in zip(examples["premise1"], examples["premise2"], examples["conclusion"])
        ]
        encoding = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors=None  # 不返回 tensor，交给 collator 处理
        )
        encoding["labels"] = examples["labels"]
        encoding["plausibility"] = examples["plausibility"]  # 保留用于后续分析（虽然 trainer 不用）
        return encoding

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["premise1", "premise2", "conclusion"]
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["premise1", "premise2", "conclusion"]
    )

    # 加载自定义 Qwen 分类模型
    model = QwenForSequenceClassification(
        model_name_or_path=cfg.model_name,
        num_labels=2,
        dropout=0.1
    ).to(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 可选：开启梯度检查点节省显存
    model.qwen.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir='/home/luorongchuan/workspace_134/Semeval2026/Q1/code/qwen2.5-1.5b-noaug/out',
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=2,
        warmup_steps=0,
        weight_decay=cfg.weight_decay,
        report_to=["none"],
        remove_unused_columns=False,  # 保留 plausibility（虽不用，但无害）
        dataloader_pin_memory=False,  # 避免 Qwen tokenizer 兼容问题
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
    model_save_path = "/home/luorongchuan/workspace_134/Semeval2026/Q1/code/qwen2.5-1.5b-noaug/out"
    os.makedirs(model_save_path, exist_ok=True)

    # 保存完整模型权重（包括 classifier）
    torch.save(model.state_dict(), os.path.join(model_save_path, "pytorch_model.bin"))

    # 可选：也保存 tokenizer 和 config
    tokenizer.save_pretrained(model_save_path)
    # 如果你想保存原始 Qwen config，也可以：
    model.qwen.config.save_pretrained(model_save_path)

    # 评估
    pred_out = trainer.predict(val_dataset)
    pred_logits = pred_out.predictions
    if isinstance(pred_logits, tuple):
        pred_logits = pred_logits[0]
    y_pred = np.argmax(pred_logits, axis=-1)
    y_true = np.array([d['labels'] for d in val_data], dtype=int)
    plaus_arr = np.array([d['plausibility'] for d in val_data], dtype=int)

    content_metrics = compute_content_effects(y_pred, y_true, plaus_arr)

    print(
        "[Content Bias Metrics] "
        f"IPCE={content_metrics['ipce']:.6f}, "
        f"CPCE={content_metrics['cpce']:.6f}, "
        f"TCE={content_metrics['tce']:.6f}, "
        f"ACC/TCE={content_metrics['acc_over_tce']:.6f}"
    )

    eval_results = trainer.evaluate()
    eval_results.update(content_metrics)
    return eval_results


# -----------------------------
# 测试集预测（适配 Qwen）
# -----------------------------

def predict_on_test_set(test_filepath: str, model_path: str, tokenizer_name: str, output_json_path: str, max_length: int = 512):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ✅ 手动实例化你的自定义模型
    model = QwenForSequenceClassification(
        model_name_or_path=tokenizer_name,  # 或者传入原始模型名，如 "Qwen/Qwen2-1.5B"
        num_labels=2
    )

    # ✅ 加载你在训练时保存的 state_dict（权重）
    state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(state_dict)

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
            continue

        p1, p2, c = safe_split_syllogism(syllogism)
        text = f"Premise 1: {p1}\nPremise 2: {p2}\nConclusion: {c}"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            pred_label = torch.argmax(outputs["logits"], dim=-1).item()

        predictions.append({"id": _id, "validity": bool(pred_label)})

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ 预测完成，结果已保存至: {output_json_path}")


# -----------------------------
# 主程序
# -----------------------------

if __name__ == "__main__":
    import os

    train_data_path = '/home/luorongchuan/workspace_134/Semeval2026/Q1/data/q1_test_merge.json'
    data = load_and_process_data(train_data_path)

    cfg = NLIConfig(
        model_name="Qwen/Qwen2.5-1.5B",  # ← Qwen 模型
        max_length=512,
        batch_size=32,                  # ← 小 batch
        epochs=5,
        learning_rate=1e-5,            # ← 小 lr
        weight_decay=0.01,
        seed=42
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    eval_results = train_model(data, cfg)
    print("Evaluation results:", eval_results)

    # 预测
    TEST_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q1/data/test_data_subtask_1.json"
    OUTPUT_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q1/code/qwen2.5-1.5b-noaug/predictions.json"
    MODEL_SAVE_DIR = "/home/luorongchuan/workspace_134/Semeval2026/Q1/code/qwen2.5-1.5b-noaug/out"

    if os.path.exists(TEST_FILE):
        predict_on_test_set(
            test_filepath=TEST_FILE,
            model_path=MODEL_SAVE_DIR,
            tokenizer_name=cfg.model_name,
            output_json_path=OUTPUT_FILE,
            max_length=cfg.max_length
        )