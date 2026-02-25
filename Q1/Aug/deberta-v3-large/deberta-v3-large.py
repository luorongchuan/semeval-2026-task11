# ========== 放在文件最顶部：外网镜像 & 可选 Token ==========

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_READ_TIMEOUT"] = "60"
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "4,5"
# 确保未开启离线模式（否则仍不会联网）
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_HUB_OFFLINE", None)

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

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
    TrainerCallback,
)

# -----------------------------
# 评估指标
# -----------------------------

def compute_metrics(p):
    logits = p.predictions
    # 可能是 (logits,) 的 tuple
    if isinstance(logits, tuple):
        logits = logits[0]
    # 也可能是按 batch 分的 list，需要拼接
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
    model_name: str = "microsoft/deberta-v3-large"
    max_length: int = 512
    entail_thresh: float = 0.5
    contra_guard: float = 0.4
    device: Optional[str] = None  # "cuda" / "cpu" / None 自动
    show_confusion: bool = True   # 评估结束是否打印混淆矩阵
    # 微调配置
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    seed:int = 42

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
# （1）数据加载：读取 plausibility / formal_validity  [NEW]
# -----------------------------

def _to_bin_bool(x):
    """把多种写法统一成 {0,1}；不识别时返回 None"""
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

        # 目标标签：validity（== formal validity），直接合并成一行
        label = _to_bin_bool(example.get('validity', 0))  # 直接获取并处理 validity

        # 只保留 plausibility（训练集有，测试集可能无）
        plaus = _to_bin_bool(example.get("plausibility"))  # 1/0/None

        processed_data.append({
            'premise1': premise1 or "",
            'premise2': premise2 or "",
            'conclusion': conclusion or "",
            'labels': label,
            'plausibility': plaus
        })
    return processed_data


# -----------------------------
# （2）计算内容效应指标的工具函数  [NEW]
# -----------------------------

import numpy as np

def compute_content_effects(preds, labels, plaus):
    """
    - IPCE:  对每个 plaus∈{0,1}，|acc(true_valid=1) - acc(true_valid=0)| 的平均
    - CPCE:  对每个 true_valid∈{0,1}，|acc(plaus=1) - acc(plaus=0)| 的平均
    - TCE :  (IPCE + CPCE) / 2
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    plaus = np.asarray(plaus)

    def _safe_acc(preds, labels, mask):
        """Helper function to safely calculate accuracy for a given mask."""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.nan  # Return NaN if no samples meet the condition
        return np.mean((preds[idx] == labels[idx]).astype(np.float32))

    # ===== IPCE =====
    diffs_ip = []
    for p in [0, 1]:
        m_p = (plaus == p)  # Filter by plausibility
        if m_p.sum() == 0:  # Skip if no items with the current plausibility
            continue
        acc_valid = _safe_acc(preds, labels, m_p & (labels == 1))  # Accuracy for valid items
        acc_invalid = _safe_acc(preds, labels, m_p & (labels == 0))  # Accuracy for invalid items
        if not np.isnan(acc_valid) and not np.isnan(acc_invalid):
            diffs_ip.append(abs(acc_valid - acc_invalid))  # Difference between valid and invalid accuracies
    ipce = np.mean(diffs_ip) if diffs_ip else np.nan  # Average difference for IPCE

    # ===== CPCE =====
    diffs_cp = []
    for fv_val in [0, 1]:
        m_fv = (labels == fv_val)  # Filter by validity
        if m_fv.sum() == 0:  # Skip if no items with the current validity
            continue
        acc_plaus = _safe_acc(preds, labels, m_fv & (plaus == 1))  # Accuracy for plausible items
        acc_implaus = _safe_acc(preds, labels, m_fv & (plaus == 0))  # Accuracy for implausible items
        if not np.isnan(acc_plaus) and not np.isnan(acc_implaus):
            diffs_cp.append(abs(acc_plaus - acc_implaus))  # Difference between plausible and implausible accuracies
    cpce = np.mean(diffs_cp) if diffs_cp else np.nan  # Average difference for CPCE

    # ===== 汇总 (Summary) =====
    overall_acc = np.mean((preds == labels).astype(np.float32))  # Overall accuracy
    tce = (ipce + cpce) / 2.0 if not np.isnan(ipce) and not np.isnan(cpce) else np.nan  # Total Content Effect

    # Handling division by zero for accuracy over TCE
    if tce < 0:
        return 0.0
    else:
        log_penalty = math.log(1 + tce)
        acc_over_tce = overall_acc / (1 + log_penalty)  # Avoid division by zero

    # Return dictionary of metrics
    return {
        "ipce": ipce,
        "cpce": cpce,
        "tce": tce,
        "accuracy": overall_acc,
        "acc_over_tce": acc_over_tce
    }
def create_stratify_labels(labels, plausibilities):
    """
    将 (label, plaus) 组合成唯一类别，用于分层。
    注意：plaus 可能为 None → 替换为 -1 或跳过（但最好保留）
    """
    stratify = []
    for y, p in zip(labels, plausibilities):
        # 处理 plausibility 为 None 的情况
        if p is None or pd.isna(p):
            p = -1  # 表示缺失
        stratify.append(f"{y}_{p}")
    return stratify



# -----------------------------
# 微调模型
# -----------------------------
def train_model(data, cfg: NLIConfig):
    # 切分数据
    labels = [d['labels'] for d in data]
    plaus = [d['plausibility'] for d in data]
    # 构建联合分层标签
    stratify_labels = create_stratify_labels(labels, plaus)
    # 检查是否所有类别都有足够样本（至少2个才能 split）
    from collections import Counter
    counts = Counter(stratify_labels)
    min_count = min(counts.values())
    if min_count < 2:
        print("⚠️ 警告：某些 (label, plaus) 组合样本数 < 2，无法严格分层。")
        # 可选：回退到仅按 label 分层
        stratify_for_split = labels
    else:
        stratify_for_split = stratify_labels

    # 执行 split
    train_data, val_data = train_test_split(
        data,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=stratify_for_split
)

    # 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # ===================== 数据集映射 =====================
    # 传递 tokenizer 给 preprocess_function
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    # print("train_dataset",train_dataset)
    # 在 map 时传递 tokenizer
    def preprocess_function(examples, tokenizer):
        # print(f"Examples: {examples}")  # 打印出 examples 的内容，查看是否有 'labels' 字段
        sep = tokenizer.sep_token or "[SEP]"
        premise = [
            f"Premise1: {p1} {sep} Premise2: {p2}".strip()
            for p1, p2 in zip(examples["premise1"], examples["premise2"])
        ]
        hypothesis = [f"Conclusion: {c}" for c in examples["conclusion"]]

        # 返回 tokenized 数据，包括 input_ids 和 labels
        encoding = tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            max_length=cfg.max_length
            # return_tensors="pt"  # 返回 PyTorch tensor 格式
        )
        # 只打印前 3 个样本的 input_ids 和 attention_mask（可读性更好）
        sample_keys = ["input_ids", "attention_mask"]
        sampled_encoding = {
            k: v[:3] if isinstance(v, list) else v for k, v in encoding.items() if k in sample_keys
        }
        print("Sample of tokenized batch (first 3):")
        for i in range(min(3, len(encoding["input_ids"]))):
            print(f"  Sample {i}: input_ids={encoding['input_ids'][i][:20]}...")  # 只看前20个token
        print(f"Keys before adding labels: {list(encoding.keys())}")

        encoding["labels"] = examples["labels"]  # 已经是 list of int

        print(f"Keys after adding labels: {list(encoding.keys())}")
        # labels 也只打前3个
        print(f"Labels (first 3): {examples['labels'][:3]}")
        return encoding

    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=["premise1", "premise2", "conclusion"])
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=["premise1", "premise2", "conclusion"])
    # 检查 dataset 中某个样本的维度
    print(train_dataset.shape)
    print(val_dataset.shape)
    # print(f"Train dataset sample input_ids shape: {np.array(train_dataset['input_ids']).shape}")
    # print(f"Train dataset sample labels shape: {np.array(train_dataset['labels']).shape}")


    # 读取可选 token（建议在 shell 里 export HF_TOKEN=...）
    hf_token = os.environ.get("HF_TOKEN", None)
    auth_kw = {"token": hf_token} if hf_token else {}

    # 加载模型
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        use_safetensors=True,  # 👈 强制使用 safetensors
        ignore_mismatched_sizes=True,
        **auth_kw
    ).to(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 设置二分类
    model.config.problem_type = "single_label_classification"
    model.config.num_labels = 2
    model.config.label2id = {"invalid": 0, "valid": 1}
    model.config.id2label = {0: "invalid", 1: "valid"}
    # 对于 BART MNLI，分类头一般为 classification_head(dense/out_proj)
    # 保险打印一下（不同版本命名可能略有差异）
    if hasattr(model, "classification_head"):
        print("最开始的模型：", model.classification_head)
        # 把输出层改成 2 类
        model.classification_head.out_proj = torch.nn.Linear(
            model.classification_head.dense.out_features, 2
        )
        print("修改后的模型：", model.classification_head)
    else:
        # 兜底：若没有该属性，则直接替换 classifier（不同架构的命名）
        if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Linear):
            in_feat = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_feat, 2)
            print("使用 classifier 作为输出层，已改为 2 类")
        else:
            print("⚠ 未找到常见分类头，请确认模型架构。")

    print("Device:", model.device)
    # 训练参数
    training_args = TrainingArguments(
        output_dir='/home/luorongchuan/workspace_134/Semeval2026/Q1/out',
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

    # 开始训练
    trainer.train()
    trainer.save_model("/home/luorongchuan/workspace_134/Semeval2026/Q1/out")

    # 评估并计算内容效应
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

    # 输出内容效应指标
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
# 新增：对测试集进行预测
# -----------------------------

def predict_on_test_set(test_filepath: str, model_path: str, tokenizer_name: str, output_json_path: str, max_length: int = 512):
    """
    对测试集进行预测，并保存 {id: ..., validity: 0/1} 的 JSON 文件。
    
    Args:
        test_filepath: 测试集 JSON 路径，格式 [{"id": "...", "syllogism": "..."}, ...]
        model_path: 训练好的模型路径（如 ./out）
        tokenizer_name: tokenizer 名称（通常和 model_name 一致）
        output_json_path: 输出 JSON 路径
        max_length: 最大序列长度
    """


    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 读取测试数据
    with open(test_filepath, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions = []

    for item in tqdm(test_data, desc="Predicting on test set"):
        syllogism = item.get("syllogism", "").strip()
        _id = item.get("id")

        if not syllogism or _id is None:
            print(f"⚠️ 跳过无效条目: {item}")
            continue

        # 分解三段论
        premise1, premise2, conclusion = safe_split_syllogism(syllogism)

        # 构造输入
        sep = tokenizer.sep_token or "[SEP]"
        premise_text = f"Premise1: {premise1} {sep} Premise2: {premise2}".strip()
        hypothesis_text = f"Conclusion: {conclusion}"

        # Tokenize
        inputs = tokenizer(
            premise_text,
            hypothesis_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_label = torch.argmax(logits, dim=-1).item()  # 0 or 1

        predictions.append({
            "id": _id,
            "validity": bool(pred_label)  # ← 转为 Python bool，json.dump 会自动转为 true/false
        })

    # 保存结果
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"✅ 预测完成，结果已保存至: {output_json_path}")


# -----------------------------
# 示例用法（修改后的主程序）
# -----------------------------
if __name__ == "__main__":
    import os

    # === 1. 训练阶段 ===
    train_data_path = '/home/luorongchuan/workspace_134/Semeval2026/Q1/data/q1_merge_aug.json'
    data = load_and_process_data(train_data_path)

    cfg = NLIConfig( 
        model_name="microsoft/deberta-v3-large",
        max_length=512,
        entail_thresh=0.5,
        contra_guard=0.4,
        device="cuda",
        batch_size=32,
        epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        seed=42
    )
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    eval_results = train_model(data, cfg)
    print("Evaluation results:", eval_results)

    # === 2. 预测阶段 ===
    TEST_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q1/data/test_data_subtask_1.json"  # ← 你的测试文件路径
    OUTPUT_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q1/data/predictions.json"

    # 模型保存路径（与 trainer.save_model 一致）
    MODEL_SAVE_DIR = "/home/luorongchuan/workspace_134/Semeval2026/Q1/out"

    if os.path.exists(TEST_FILE):
        predict_on_test_set(
            test_filepath=TEST_FILE,
            model_path=MODEL_SAVE_DIR,
            tokenizer_name=cfg.model_name,
            output_json_path=OUTPUT_FILE,
            max_length=cfg.max_length
        )
    else:
        print(f"⚠️ 测试文件不存在: {TEST_FILE}，跳过预测。")
