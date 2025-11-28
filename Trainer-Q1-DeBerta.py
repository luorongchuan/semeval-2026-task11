import os

# ========== 环境设置 ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_READ_TIMEOUT"] = "60"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# ========== 导入依赖 ==========
import re
import math
import json
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model


# ========== 工具函数 ==========
_SENT_SPLIT = re.compile(r'\s*(?<=[\.\?!。；;])\s+')

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
            j = (i + 1 + min(cand2)) if cand2 else -1
        return i, j

    i, j = _first_two_delims(text)
    p1 = text[:i].strip() if i != -1 else text
    p2 = text[i+1:j].strip() if j != -1 else ""
    c = text[j+1:].strip() if j != -1 else ""
    return p1, p2, c


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


# ========== 配置类 ==========
@dataclass
class NLIConfig:
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    max_length: int = 512
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42


# ========== 数据加载 ==========
def load_and_process_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    processed = []
    for ex in data:
        p1, p2, c = safe_split_syllogism(ex.get('syllogism', ''))
        label = _to_bin_bool(ex.get('validity'))
        plaus = _to_bin_bool(ex.get('plausibility'))

        # 只保留标签有效的样本用于训练
        if label is not None:
            processed.append({
                'premise1': p1 or "",
                'premise2': p2 or "",
                'conclusion': c or "",
                'labels': label,
                'plausibility': plaus if plaus is not None else -1  # -1 表示缺失
            })
    return processed


# ========== 评估指标 ==========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    if isinstance(logits, list):
        logits = np.concatenate(logits, axis=0)
    preds = np.argmax(logits, axis=-1)
    return {"eval_accuracy": accuracy_score(labels, preds)}


# ========== 内容效应计算 ==========
def compute_content_effects(preds, labels, plaus):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    plaus = np.asarray(plaus)

    # 过滤 plausibility 缺失项（-1）
    valid_mask = (plaus != -1)
    if not np.any(valid_mask):
        return {
            "ipce": np.nan,
            "cpce": np.nan,
            "tce": np.nan,
            "accuracy": np.mean(preds == labels),
            "acc_over_tce": np.nan
        }

    plaus = plaus[valid_mask]
    labels = labels[valid_mask]
    preds = preds[valid_mask]

    def _safe_acc(p, l, mask):
        idx = np.where(mask)[0]
        return np.mean(p[idx] == l[idx]) if len(idx) > 0 else np.nan

    # IPCE
    diffs_ip = []
    for p_val in [0, 1]:
        m = (plaus == p_val)
        if m.sum() == 0:
            continue
        acc_v = _safe_acc(preds, labels, m & (labels == 1))
        acc_iv = _safe_acc(preds, labels, m & (labels == 0))
        if not (np.isnan(acc_v) or np.isnan(acc_iv)):
            diffs_ip.append(abs(acc_v - acc_iv))
    ipce = np.mean(diffs_ip) if diffs_ip else np.nan

    # CPCE
    diffs_cp = []
    for fv in [0, 1]:
        m = (labels == fv)
        if m.sum() == 0:
            continue
        acc_p = _safe_acc(preds, labels, m & (plaus == 1))
        acc_i = _safe_acc(preds, labels, m & (plaus == 0))
        if not (np.isnan(acc_p) or np.isnan(acc_i)):
            diffs_cp.append(abs(acc_p - acc_i))
    cpce = np.mean(diffs_cp) if diffs_cp else np.nan

    overall_acc = np.mean(preds == labels)
    tce = (ipce + cpce) / 2.0 if not (np.isnan(ipce) or np.isnan(cpce)) else np.nan
    acc_over_tce = overall_acc / (1 + math.log(1 + tce)) if not np.isnan(tce) and tce >= 0 else np.nan

    return {
        "ipce": ipce,
        "cpce": cpce,
        "tce": tce,
        "accuracy": overall_acc,
        "acc_over_tce": acc_over_tce
    }


# ========== 训练主函数 ==========
def train_model(data, cfg: NLIConfig):
    if not data:
        raise ValueError("No valid training data found (all labels are None).")

    # 划分数据集
    train_data, val_data = train_test_split(
        data,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=[d['labels'] for d in data]
    )

    # 加载模型和 tokenizer
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        use_fast=False,  # 更兼容大模型
        token=hf_token
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=True,
        token=hf_token
    )

    # 数据预处理
    def preprocess_function(examples):
        sep = tokenizer.sep_token or "[SEP]"
        premises = [
            f"Premise1: {p1} {sep} Premise2: {p2}".strip()
            for p1, p2 in zip(examples["premise1"], examples["premise2"])
        ]
        conclusions = [f"Conclusion: {c}" for c in examples["conclusion"]]
        encodings = tokenizer(
            premises,
            conclusions,
            truncation=True,
            padding=True,
            max_length=cfg.max_length,
            return_tensors="pt"
        )
        encodings["labels"] = torch.tensor(examples["labels"], dtype=torch.long)
        return encodings

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["premise1", "premise2", "conclusion", "plausibility"]
    )
    val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["premise1", "premise2", "conclusion", "plausibility"]
    )

    # LoRA 配置（修正为 DeBERTa 正确模块名）
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=16,
        lora_alpha=32,
        target_modules=["query", "key", "value", "dense"],  # DeBERTa v2/v3 的正确模块
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    # 训练参数
    training_args = TrainingArguments(
        output_dir='/home/luorongchuan/workspace_134/data/semeval2026/out',
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        weight_decay=cfg.weight_decay,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
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

    print("🚀 开始训练模型...")
    trainer.train()

    # 评估
    eval_results = trainer.evaluate()
    pred_out = trainer.predict(val_dataset)
    y_pred = np.argmax(pred_out.predictions, axis=-1)
    y_true = np.array([d["labels"] for d in val_data])
    plaus_arr = np.array([d["plausibility"] for d in val_data])

    content_metrics = compute_content_effects(y_pred, y_true, plaus_arr)

    # 打印结果
    print("\n" + "=" * 60)
    print("✅ 训练完成！评估结果：")
    print("=" * 60)
    print(f"🔹 准确率 (Accuracy):           {content_metrics['accuracy']:.4f}")
    print(f"🔹 IPCE (信念干扰):             {content_metrics['ipce']:.4f}" if not np.isnan(content_metrics['ipce']) else "🔹 IPCE: N/A")
    print(f"🔹 CPCE (逻辑干扰):             {content_metrics['cpce']:.4f}" if not np.isnan(content_metrics['cpce']) else "🔹 CPCE: N/A")
    print(f"🔹 TCE (总内容效应):            {content_metrics['tce']:.4f}" if not np.isnan(content_metrics['tce']) else "🔹 TCE: N/A")
    print(f"🔹 Accuracy / (1 + log(1+TCE)): {content_metrics['acc_over_tce']:.4f}" if not np.isnan(content_metrics['acc_over_tce']) else "🔹 Acc/TCE: N/A")
    print("=" * 60)

    # 合并结果
    final_results = {
        "eval_accuracy": content_metrics["accuracy"],
        "ipce": float(content_metrics["ipce"]) if not np.isnan(content_metrics["ipce"]) else None,
        "cpce": float(content_metrics["cpce"]) if not np.isnan(content_metrics["cpce"]) else None,
        "tce": float(content_metrics["tce"]) if not np.isnan(content_metrics["tce"]) else None,
        "acc_over_tce": float(content_metrics["acc_over_tce"]) if not np.isnan(content_metrics["acc_over_tce"]) else None,
    }
    return final_results


# ========== 主程序 ==========
if __name__ == "__main__":
    # 固定随机种子
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 加载数据
    data_path = '/home/luorongchuan/workspace_134/data/semeval2026/train_data/merged_train_data.json'
    data = load_and_process_data(data_path)
    print(f"✅ 加载 {len(data)} 条有效训练样本")

    # 配置
    cfg = NLIConfig(
        model_name="microsoft/deberta-v2-xlarge-mnli",
        max_length=512,
        batch_size=32,
        epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=seed
    )

    # 训练
    results = train_model(data, cfg)
    print("\n最终结果字典：")
    print(results)