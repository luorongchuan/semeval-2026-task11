# ========== 放在文件最顶部：外网镜像 & 可选 Token ==========
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_READ_TIMEOUT"] = "60"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# -----------------------------
# 工具函数
# -----------------------------

_SENT_SPLIT = re.compile(r'\s*(?<=[\.\?!。；;])\s+')


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


def safe_split_syllogism(s: str) -> Tuple[str, str, str]:
    text = s.strip()
    if not text:
        return "", "", ""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], parts[1], ""
    elif len(parts) == 1:
        return parts[0], "", ""
    else:
        return "", "", ""


def create_stratify_col(labels: np.ndarray, plaus: np.ndarray) -> List[str]:
    return [f"{y}_{p if p in (0, 1) else -1}" for y, p in zip(labels, plaus)]


def compute_content_effects(preds, labels, plaus):
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    plaus = np.asarray(plaus)

    def _safe_acc(preds, labels, mask):
        idx = np.where(mask)[0]
        return np.nan if len(idx) == 0 else np.mean((preds[idx] == labels[idx]).astype(float))

    # IPCE
    diffs_ip = []
    for p_val in [0, 1]:
        m = (plaus == p_val)
        if m.sum() == 0:
            continue
        acc_valid = _safe_acc(preds, labels, m & (labels == 1))
        acc_invalid = _safe_acc(preds, labels, m & (labels == 0))
        if not (np.isnan(acc_valid) or np.isnan(acc_invalid)):
            diffs_ip.append(abs(acc_valid - acc_invalid))
    ipce = np.mean(diffs_ip) if diffs_ip else np.nan

    # CPCE
    diffs_cp = []
    for fv_val in [0, 1]:
        m = (labels == fv_val)
        if m.sum() == 0:
            continue
        acc_plaus = _safe_acc(preds, labels, m & (plaus == 1))
        acc_implaus = _safe_acc(preds, labels, m & (plaus == 0))
        if not (np.isnan(acc_plaus) or np.isnan(acc_implaus)):
            diffs_cp.append(abs(acc_plaus - acc_implaus))
    cpce = np.mean(diffs_cp) if diffs_cp else np.nan

    overall_acc = np.mean((preds == labels).astype(float))
    tce = (ipce + cpce) / 2.0 if not (np.isnan(ipce) or np.isnan(cpce)) else np.nan

    if np.isnan(tce) or tce <= 0:
        acc_over_tce = overall_acc
    else:
        log_penalty = math.log(1 + tce)
        acc_over_tce = overall_acc / (1 + log_penalty)

    return {
        "ipce": None if np.isnan(ipce) else float(ipce),
        "cpce": None if np.isnan(cpce) else float(cpce),
        "tce": None if np.isnan(tce) else float(tce),
        "accuracy": float(overall_acc),
        "acc_over_tce": float(acc_over_tce),
    }


# -----------------------------
# 配置类
# -----------------------------

@dataclass
class NLIConfig:
    model_name: str = "facebook/bart-large-mnli"
    max_length: int = 512
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    n_splits: int = 5
    out_dir_prefix: str = "/home/luorongchuan/workspace_134/data/semeval2026"
    device: Optional[str] = None  

# -----------------------------
# 数据加载
# -----------------------------

def load_and_process_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    processed = []
    for ex in data:
        p1, p2, c = safe_split_syllogism(ex.get('syllogism', ''))
        label = _to_bin_bool(ex.get('validity', 0))
        plaus = _to_bin_bool(ex.get("plausibility"))
        processed.append({
            'premise1': p1 or "",
            'premise2': p2 or "",
            'conclusion': c or "",
            'labels': label,
            'plausibility': plaus
        })
    return processed


# -----------------------------
# 单折训练函数
# -----------------------------

def train_one_fold(train_df, val_df, cfg: NLIConfig, fold_id: int):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess_function(examples):
        sep = tokenizer.sep_token or "[SEP]"
        premise = [
            f"Premise1: {p1} {sep} Premise2: {p2}".strip()
            for p1, p2 in zip(examples["premise1"], examples["premise2"])
        ]
        hypothesis = [f"Conclusion: {c}" for c in examples["conclusion"]]
        encoding = tokenizer(
            premise, hypothesis,
            padding=True,
            truncation=True,
            max_length=cfg.max_length
        )
        encoding["labels"] = examples["labels"]
        return encoding

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    train_ds = train_ds.map(preprocess_function, batched=True,
                            remove_columns=["premise1", "premise2", "conclusion", "plausibility"])
    val_ds = val_ds.map(preprocess_function, batched=True,
                        remove_columns=["premise1", "premise2", "conclusion", "plausibility"])

    hf_token = os.environ.get("HF_TOKEN")
    auth_kw = {"token": hf_token} if hf_token else {}

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
        **auth_kw
    ).to(device)

    model.config.problem_type = "single_label_classification"
    model.config.label2id = {"invalid": 0, "valid": 1}
    model.config.id2label = {0: "invalid", 1: "valid"}

    # 修改分类头
    if hasattr(model, "classification_head"):
        model.classification_head.out_proj = torch.nn.Linear(
            model.classification_head.dense.out_features, 2
        )
    elif hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
        in_feat = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_feat, 2)

    out_dir = os.path.join(cfg.out_dir_prefix, f'out_fold{fold_id}')
    os.makedirs(out_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=out_dir,
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
        seed=cfg.seed + fold_id,
        dataloader_num_workers=2,
    )

    def compute_metrics(p):
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(logits, axis=-1)
        return {"eval_accuracy": accuracy_score(p.label_ids, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    pred_out = trainer.predict(val_ds)
    val_logits = pred_out.predictions[0] if isinstance(pred_out.predictions, tuple) else pred_out.predictions
    val_labels = val_df["labels"].to_numpy(dtype=int)
    fold_acc = accuracy_score(val_labels, np.argmax(val_logits, axis=-1))
    print(f"[Fold {fold_id}] Val Accuracy: {fold_acc:.6f}")

    return val_logits, val_labels


# -----------------------------
# 5 折交叉验证主函数
# -----------------------------

def train_5fold_cv(data: List[Dict], cfg: NLIConfig):
    df = pd.DataFrame(data)
    labels = df["labels"].values
    plaus = df["plausibility"].values

    stratify_col = create_stratify_col(labels, plaus)
    counts = Counter(stratify_col)
    if min(counts.values()) < 2:
        print("⚠️ 警告：某些 (label, plaus) 组合样本数 < 2，回退到仅按 label 分层")
        stratify_for_split = labels
    else:
        stratify_for_split = stratify_col

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    indices = np.arange(len(df))

    oof_logits = np.zeros((len(df), 2), dtype=np.float32)
    fold_accuracies = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(indices, stratify_for_split), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        val_logits, val_labels = train_one_fold(train_df, val_df, cfg, fold_id)
        oof_logits[val_idx] = val_logits
        fold_acc = accuracy_score(val_labels, np.argmax(val_logits, axis=-1))
        fold_accuracies.append(fold_acc)

    # OOF 评估
    y_true = df["labels"].to_numpy(dtype=int)
    y_pred = np.argmax(oof_logits, axis=-1)
    plaus_arr = np.array([p if p in (0, 1) else -1 for p in df["plausibility"].tolist()], dtype=int)

    oof_acc = accuracy_score(y_true, y_pred)
    content_metrics = compute_content_effects(y_pred, y_true, plaus_arr)

    print("\n" + "=" * 50)
    print("✅ 5-Fold Cross-Validation Results (OOF)")
    print("=" * 50)
    print(f"OOF Accuracy          : {oof_acc:.6f}")
    print(f"Mean Fold Accuracy    : {np.mean(fold_accuracies):.6f}")
    print(f"Fold Accuracies       : {[f'{a:.6f}' for a in fold_accuracies]}")
    print(f"IPCE                  : {content_metrics['ipce']:.6f}")
    print(f"CPCE                  : {content_metrics['cpce']:.6f}")
    print(f"TCE                   : {content_metrics['tce']:.6f}")
    print(f"ACC / TCE             : {content_metrics['acc_over_tce']:.6f}")
    print("=" * 50)

    return {
        "oof_accuracy": float(oof_acc),
        "mean_fold_accuracy": float(np.mean(fold_accuracies)),
        "fold_accuracies": [float(a) for a in fold_accuracies],
        "ipce": content_metrics["ipce"],
        "cpce": content_metrics["cpce"],
        "tce": content_metrics["tce"],
        "acc_over_tce": content_metrics["acc_over_tce"],
        "oof_logits": oof_logits,
    }


# -----------------------------
# 主程序入口
# -----------------------------

if __name__ == "__main__":
    train_json = "/home/luorongchuan/workspace_134/data/semeval2026/train_data/merged_train_data.json"
    cfg = NLIConfig(
        model_name="facebook/bart-large-mnli",
        max_length=512,
        batch_size=32,
        epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42,
        n_splits=5,
        out_dir_prefix="/home/luorongchuan/workspace_134/data/semeval2026",
    )

    # 固定随机种子
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # 加载数据
    data = load_and_process_data(train_json)

    # 执行 5 折交叉验证
    results = train_5fold_cv(data, cfg)

    # 可选：保存结果到 JSON
    import json
    with open(os.path.join(cfg.out_dir_prefix, "cv_results.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if k != "oof_logits"}, f, indent=4, ensure_ascii=False)

    print("\n🎉 5-Fold CV completed. Results saved.")