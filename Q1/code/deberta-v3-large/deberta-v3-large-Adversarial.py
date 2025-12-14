# ========== 放在文件最顶部：外网镜像 & 可选 Token ==========

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_READ_TIMEOUT"] = "120"
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
    AutoModel,
    
)
from transformers import AutoConfig, DebertaV2ForSequenceClassification, DebertaV2Tokenizer
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
# -----------------------------
# 自定义模型：主任务 + 对抗任务
# -----------------------------

from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.lambda_ * grad_output.neg(), None

class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class DebertaWithAdversary(torch.nn.Module):
    def __init__(self, base_model_name, num_labels=2, adv_hidden_dim=128, lambda_adv=0.05, **from_pretrained_kwargs):
        super().__init__()
        self.deberta = AutoModel.from_pretrained(base_model_name, **from_pretrained_kwargs,use_safetensors=True)
        hidden_size = self.deberta.config.hidden_size
        
        self.num_labels = num_labels  # ← 关键：用于 forward 中的 CrossEntropyLoss
        self.classifier = torch.nn.Linear(hidden_size, num_labels)
        self.adversary = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, adv_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(adv_hidden_dim, 1)
        )
        self.grl = GradientReversalLayer(lambda_=lambda_adv)

    def forward(self, input_ids, attention_mask=None, labels=None, plaus_labels=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = outputs.last_hidden_state[:, 0, :]  # [B, H]

        # 主任务
        logits = self.classifier(cls_repr)
        main_loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            main_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 对抗任务：通过 GRL
        cls_repr_grl = self.grl(cls_repr)  # ← 梯度在此处反转！
        adv_logits = self.adversary(cls_repr_grl).squeeze(-1)
        adv_loss = None
        if plaus_labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            adv_loss = loss_fct(adv_logits, plaus_labels.float())

        return SequenceClassifierOutput(
            loss=main_loss,
            logits=logits,
        ), adv_loss


# -----------------------------
# 自定义 Trainer：支持对抗损失
# -----------------------------

class AdversarialTrainer(Trainer):
    def __init__(self, lambda_adv=0.05, **kwargs):
        super().__init__(**kwargs)
        self.lambda_adv = lambda_adv

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs, adv_loss = model(**inputs)
        main_loss = outputs.loss
        total_loss = main_loss + adv_loss  # 注意：GRL 已处理梯度方向，此处直接相加
        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # 移除 plaus_labels（评估不需要）
        has_labels = "labels" in inputs
        if not has_labels:
            raise ValueError("Evaluation requires labels.")

        # 确保不传 plaus_labels
        eval_inputs = {k: v for k, v in inputs.items() if k != "plaus_labels"}

        # 手动前向（不依赖父类）
        model.eval()
        with torch.no_grad():
            # 显式传 plaus_labels=None，避免 forward 报错
            outputs, _ = model(**eval_inputs, plaus_labels=None)

        if prediction_loss_only:
            return (outputs.loss, None, None)

        return (outputs.loss, outputs.logits, inputs["labels"])


# -----------------------------
# 微调模型（含对抗去偏）
# -----------------------------
def train_model(data, cfg: NLIConfig):
    # 过滤掉 plausibility 为 None 的样本（对抗训练需要真实标签）
    filtered_data = []
    for d in data:
        if d['labels'] is not None and d['plausibility'] is not None:
            filtered_data.append(d)
    print(f"Filtered data: {len(filtered_data)} / {len(data)} samples with both labels & plausibility.")

    # 切分数据
    labels = [d['labels'] for d in filtered_data]
    plaus = [d['plausibility'] for d in filtered_data]  # 已确保非 None

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
        filtered_data,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=stratify_for_split
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess_function(examples):
        sep = tokenizer.sep_token or "[SEP]"
        premise = [
            f"Premise1: {p1} {sep} Premise2: {p2}".strip()
            for p1, p2 in zip(examples["premise1"], examples["premise2"])
        ]
        hypothesis = [f"Conclusion: {c}" for c in examples["conclusion"]]

        encoding = tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            max_length=cfg.max_length
        )
        encoding["labels"] = examples["labels"]
        encoding["plaus_labels"] = examples["plausibility"]  # 0/1
        return encoding

    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["premise1", "premise2", "conclusion"])
    val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["premise1", "premise2", "conclusion"])

    hf_token = os.environ.get("HF_TOKEN", None)
    auth_kw = {"token": hf_token} if hf_token else {}

    # 初始化带对抗头的模型（✅ 修正：移除 use_safetensors，透传合法参数）
    model = DebertaWithAdversary(
        base_model_name=cfg.model_name,
        num_labels=2,
        lambda_adv=0.05,
        ignore_mismatched_sizes=True,  # 允许分类头尺寸不匹配（因为我们自定义了 classifier）
        **auth_kw
    ).to(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

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
        seed=cfg.seed,
    )

    trainer = AdversarialTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        lambda_adv=0.05
    )

    trainer.train()

    # === 完整保存：config + tokenizer + safetensors 权重（仅主干+classifier）===
    output_dir = "/home/luorongchuan/workspace_134/Semeval2026/Q1/out"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存 tokenizer
    tokenizer.save_pretrained(output_dir)

    # 2. 保存 config.json（来自原始 DeBERTa）
    model.deberta.config.save_pretrained(output_dir)

    # 3. 保存过滤后的权重为 safetensors
    state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("adversary.")  # 移除对抗头
    }

    # 注意：AutoModelForSequenceClassification 需要 classifier 命名为 "classifier"
    # 而你的模型中是 self.classifier，这没问题，但需确保加载时结构匹配
    from safetensors.torch import save_file
    save_file(filtered_state_dict, os.path.join(output_dir, "model.safetensors"))

    print(f"✅ 模型 config、tokenizer 和 safetensors 已保存到: {output_dir}")

    # 评估
    eval_results = trainer.evaluate()

    pred_out = trainer.predict(val_dataset)
    pred_logits = pred_out.predictions
    if isinstance(pred_logits, tuple):
        pred_logits = pred_logits[0]
    y_pred = np.argmax(pred_logits, axis=-1)
    y_true = np.array([d['labels'] for d in val_data])
    plaus_arr = np.array([d['plausibility'] for d in val_data])

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
# 新增：对测试集进行预测（保持不变）
# -----------------------------

def predict_on_test_set(test_filepath: str, model_path: str, output_json_path: str, max_length: int = 512):
    from safetensors.torch import load_file

    # 1. 加载 tokenizer（显式指定类，避免 AutoTokenizer bug）
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_path)

    # 2. 加载 config 并指定 num_labels（关键！）
    config = AutoConfig.from_pretrained(model_path, num_labels=2)
    model = DebertaV2ForSequenceClassification(config)

    # 3. 加载权重（只加载一次，strict=False 更安全）
    state_dict = load_file(os.path.join(model_path, "model.safetensors"))
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print("⚠️ Missing keys in state dict:", missing_keys)
    if unexpected_keys:
        print("⚠️ Unexpected keys in state dict:", unexpected_keys)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)

    with open(test_filepath, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    predictions = []
    for item in tqdm(test_data, desc="Predicting on test set"):
        syllogism = item.get("syllogism", "").strip()
        _id = item.get("id")
        if not syllogism or _id is None:
            continue

        premise1, premise2, conclusion = safe_split_syllogism(syllogism)
        sep = tokenizer.sep_token or "[SEP]"
        premise_text = f"Premise1: {premise1} {sep} Premise2: {premise2}".strip()
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
            pred_label = torch.argmax(outputs.logits, dim=-1).item()

        predictions.append({
            "id": _id,
            "validity": bool(pred_label)
        })

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
    TEST_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q1/data/test_data_subtask_1.json"
    OUTPUT_FILE = "/home/luorongchuan/workspace_134/Semeval2026/Q1/data/predictions.json"
    MODEL_SAVE_DIR = "/home/luorongchuan/workspace_134/Semeval2026/Q1/out"

    if os.path.exists(TEST_FILE):
        predict_on_test_set(
            test_filepath=TEST_FILE,
            model_path=MODEL_SAVE_DIR,
            output_json_path=OUTPUT_FILE,
            max_length=cfg.max_length
        )
    else:
        print(f"⚠️ 测试文件不存在: {TEST_FILE}，跳过预测。")
