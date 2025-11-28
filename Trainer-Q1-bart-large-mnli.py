# ==============================
# 🧠 Syllogism Validity Classifier with Content Bias Evaluation
# ==============================
# ------------------------------
# 🔧 配置与常量
# ------------------------------

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "5"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import re
import math
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# 尝试导入 rich 用于美化输出（非必需）
try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

_SENT_SPLIT = re.compile(r'\s*(?<=[\.\?!。；;])\s+')


@dataclass
class NLIConfig:
    model_name: str = "facebook/bart-large-mnli"
    max_length: int = 512
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 42
    device: Optional[str] = None  # 自动检测


# ------------------------------
# 📥 数据加载与预处理
# ------------------------------

def safe_split_syllogism(s: str) -> Tuple[str, str, str]:
    """将三段论文本安全拆分为两个前提和一个结论"""
    text = s.strip()
    if not text:
        return "", "", ""
    parts = [p.strip() for p in _SENT_SPLIT.split(text) if p.strip()]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]

    def _first_two_delims(t: str) -> Tuple[int, int]:
        cand = [t.find(ch) for ch in ['.', '。'] if t.find(ch) != -1]
        i = min(cand) if cand else -1
        j = -1
        if i != -1:
            rest = t[i + 1:]
            cand2 = [rest.find(ch) for ch in ['.', '。'] if rest.find(ch) != -1]
            j = (i + 1 + min(cand2)) if cand2 else -1
        return i, j

    i, j = _first_two_delims(text)
    p1 = text[:i].strip() if i != -1 else text
    p2 = text[i + 1:j].strip() if j != -1 else ""
    c = text[j + 1:].strip() if j != -1 else ""
    return p1, p2, c


def _to_bin_bool(x) -> Optional[int]:
    """将多种格式的标签转为 0/1"""
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


def load_and_process_data(filepaths: List[str]) -> List[Dict[str, Any]]:
    """加载并预处理多个 JSON 数据集"""
    all_data = []
    for filepath in filepaths:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        processed = []
        for ex in data:
            p1, p2, c = safe_split_syllogism(ex.get('syllogism', ''))
            label = _to_bin_bool(ex.get('validity')) 
            plaus = _to_bin_bool(ex.get('plausibility'))
            # plaus = plaus if plaus is not None else -1  # -1 表示缺失
            processed.append({
                'premise1': p1,
                'premise2': p2,
                'conclusion': c,
                'labels': label,
                'plausibility': plaus
            })
        all_data.extend(processed)
        print(f"✅ Loaded {len(processed)} examples from {filepath}")
    return all_data


def load_test_data(filepath: str) -> List[Dict[str, Any]]:
    """单独加载测试数据（不用于训练）"""
    return load_and_process_data([filepath])

def set_all_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------------------------------
# 📊 评估指标（与评估脚本严格对齐）
# ------------------------------

def calculate_smooth_combined_metric(overall_accuracy: float, total_content_effect: float) -> float:
    """
    Computes a smooth combined score using the natural logarithm.
    Formula: accuracy / (1 + ln(1 + content_effect))
    Input: both in percentage (e.g., 85.0 for 85%)
    Output: score in [0, 100]
    """
    if total_content_effect < 0:
        return 0.0
    log_penalty = math.log(1 + total_content_effect)
    combined_smooth_score = overall_accuracy / (1 + log_penalty)
    return combined_smooth_score


def compute_content_effects(preds, labels, plaus) -> Dict[str, Any]:
    """
    计算内容效应指标，与评估脚本完全一致。
    - 所有 accuracy 和 bias 均为百分数（0~100）
    - plaus 中 -1 视为缺失，自动过滤
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    # plaus = np.where(np.asarray(plaus) == -1, np.nan, plaus).astype(np.float32)
    plaus = np.asarray(plaus, dtype=int)
    # 安全检查：确保 plaus 只有 0 和 1
    if not np.all((plaus == 0) | (plaus == 1)):
        invalid_vals = np.unique(plaus[~((plaus == 0) | (plaus == 1))])
        raise ValueError(f"Invalid plausibility values found: {invalid_vals}. Only 0 and 1 are allowed.")

    def _acc(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.nan, 0
        acc_val = float(np.mean(preds[idx] == labels[idx]) * 100)
        return acc_val, len(idx)

    acc_pv, n_pv = _acc((labels == 1) & (plaus == 1))   # Valid & Plausible
    acc_iv, n_iv = _acc((labels == 1) & (plaus == 0))   # Valid & Implausible
    acc_pi, n_pi = _acc((labels == 0) & (plaus == 1))   # Invalid & Plausible
    acc_ii, n_ii = _acc((labels == 0) & (plaus == 0))   # Invalid & Implausible

    overall_acc = float(np.mean(preds == labels) * 100)

    def safe_diff(a, b):
        if np.isnan(a) or np.isnan(b):
            return 0.0
        return abs(float(a) - float(b))

    intra_valid_diff = safe_diff(acc_pv, acc_iv)
    intra_invalid_diff = safe_diff(acc_pi, acc_ii)
    content_effect_intra = (intra_valid_diff + intra_invalid_diff) / 2.0

    inter_plausible_diff = safe_diff(acc_pv, acc_pi)
    inter_implausible_diff = safe_diff(acc_iv, acc_ii)
    content_effect_inter = (inter_plausible_diff + inter_implausible_diff) / 2.0

    tot_content_effect = (content_effect_intra + content_effect_inter) / 2.0

    combined_smooth_score = calculate_smooth_combined_metric(overall_acc, tot_content_effect)

    return {
        "accuracy": overall_acc,
        "acc_plausible_valid": acc_pv if not np.isnan(acc_pv) else 0.0,
        "acc_implausible_valid": acc_iv if not np.isnan(acc_iv) else 0.0,
        "acc_plausible_invalid": acc_pi if not np.isnan(acc_pi) else 0.0,
        "acc_implausible_invalid": acc_ii if not np.isnan(acc_ii) else 0.0,
        "content_effect_intra_validity_label": content_effect_intra,
        "content_effect_inter_validity_label": content_effect_inter,
        "tot_content_effect": tot_content_effect,
        "combined_smooth_score": combined_smooth_score,
        "counts": {"vp": n_pv, "vi": n_iv, "ivp": n_pi, "ivi": n_ii}
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    if torch.is_tensor(logits):
        logits = logits.detach().cpu().numpy()
    elif not isinstance(logits, np.ndarray):
        logits = np.array(logits)
        if logits.dtype == object:
            logits = np.vstack(logits)
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    return {"eval_accuracy": acc}


# ------------------------------
# 🚀 模型训练与评估
# ------------------------------

def train_model(train_data: List[Dict], test_data: List[Dict], cfg: NLIConfig) -> Dict[str, float]:
    # ====== 1. 划分训练集和验证集（新增！）======
    train_df = pd.DataFrame(train_data)
    
    # 打乱并划分（保留标签分布可选，这里简单随机）
    train_df = train_df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    val_ratio = 0.2  # 可配置为 cfg.val_ratio
    val_size = int(len(train_df) * val_ratio)
    
    val_df = train_df[:val_size]
    train_df = train_df[val_size:]
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # ====== 2. Tokenizer & Dataset ======
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess(examples):
        sep = tokenizer.sep_token or "[SEP]"
        premises = [f"Premise1: {p1} {sep} Premise2: {p2}" for p1, p2 in zip(examples["premise1"], examples["premise2"])]
        conclusions = [f"Conclusion: {c}" for c in examples["conclusion"]]
        enc = tokenizer(premises, conclusions, padding=True, truncation=True, max_length=cfg.max_length)
        enc["labels"] = examples["labels"]
        return enc

    train_ds = train_ds.map(preprocess, batched=True, remove_columns=["premise1", "premise2", "conclusion"])
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=["premise1", "premise2", "conclusion"])
    # 测试集处理
    test_df = pd.DataFrame(test_data)
    test_ds = Dataset.from_pandas(test_df).map(preprocess, batched=True, remove_columns=["premise1", "premise2", "conclusion"])

    # ====== 3. 模型 & 设备 ======
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU.")
        device = "cpu"

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,
    ).to(device)
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
    # ====== 4. 训练器 ======
    training_args = TrainingArguments(
        output_dir="./syllogism_out",
        eval_strategy="epoch",  # 训练中评估
        logging_strategy="epoch",
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=2,
        weight_decay=cfg.weight_decay,
        report_to="none",
        seed=cfg.seed,
        disable_tqdm=False,
        load_best_model_at_end=True,   # 如果不用早停，可设为 False；若要用，需配合 save_strategy
        # 如果想用早停，需设置：
        save_strategy="epoch",
        metric_for_best_model="eval_accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds, 
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # ====== 5. 训练 ======
    print("🚀 Starting training on augmented data...")
    trainer.train()

    # ====== 6. 在原始测试集上评估 ======
    print("🔍 Evaluating on original test data...")
    pred_out = trainer.predict(test_ds)
    predictions = pred_out.predictions
    # y_true = pred_out.label_ids  # ← 直接从 predict 输出拿，保证对齐！

    if isinstance(predictions, (list, tuple)):
        predictions = predictions[0]
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    elif isinstance(predictions, np.ndarray) and predictions.dtype == object:
        predictions = np.vstack(predictions.tolist())
    elif not isinstance(predictions, np.ndarray):
        try:
            predictions = np.array(predictions)
            if predictions.dtype == object:
                predictions = np.vstack(predictions)
        except Exception as e:
            raise ValueError(f"Failed to convert predictions: {e}")

    y_pred = np.argmax(predictions, axis=-1)
    y_true = test_df["labels"].values
    plaus_arr = test_df["plausibility"].replace(-1, np.nan).values

    content_metrics = compute_content_effects(y_pred, y_true, plaus_arr)

    results = {
        "accuracy": content_metrics["accuracy"],
        "acc_plausible_valid": content_metrics["acc_plausible_valid"],
        "acc_implausible_valid":content_metrics["acc_implausible_valid"] ,
        "acc_plausible_invalid": content_metrics["acc_plausible_invalid"],
        "acc_implausible_invalid": content_metrics["acc_implausible_invalid"],
        "tot_content_effect": content_metrics["tot_content_effect"],
        "combined_smooth_score": content_metrics["combined_smooth_score"],
        "ipce": content_metrics["content_effect_intra_validity_label"],
        "cpce": content_metrics["content_effect_inter_validity_label"],
    }

    _print_evaluation_results(results, content_metrics["counts"])
    return results


def _print_evaluation_results(results: Dict, counts: Dict):
    """美观打印评估结果"""
    metrics = {
        "Accuracy (%)": f"{results['accuracy']:.2f}",
        "acc_plausible_valid": f"{results['acc_plausible_valid']:.4f}",
        "acc_implausible_valid": f"{results['acc_implausible_valid']:.4f}",
        "acc_plausible_invalid": f"{results['acc_plausible_invalid']:.4f}",
        "acc_implausible_invalid": f"{results['acc_implausible_invalid']:.4f}",
        "Intra-Validity Bias (IPCE)": f"{results['ipce']:.4f}",
        "Inter-Validity Bias (CPCE)": f"{results['cpce']:.4f}",
        "Total Content Effect (TCE)": f"{results['tot_content_effect']:.4f}",
        "Smoothed Score (ACC / (1+ln(1+TCE)))": f"{results['combined_smooth_score']:.4f}",
    }

    if RICH_AVAILABLE:
        table = Table(title="📊 Model Evaluation Results", title_style="bold cyan")
        table.add_column("Metric", style="bold green")
        table.add_column("Value", style="yellow")
        for k, v in metrics.items():
            table.add_row(k, v)
        console.print(table)

        count_table = Table(title="🧮 Plausibility Group Counts", title_style="bold magenta")
        count_table.add_column("Group", style="bold blue")
        count_table.add_column("Count", style="red")
        for group, cnt in counts.items():
            name = {"vp": "Valid+Plausible", "vi": "Valid+Implausible",
                    "ivp": "Invalid+Plausible", "ivi": "Invalid+Implausible"}.get(group, group)
            count_table.add_row(name, str(cnt))
        console.print(count_table)
    else:
        print("\n📊 Model Evaluation Results:")
        print("-" * 50)
        for k, v in metrics.items():
            print(f"{k:<35}: {v}")
        print("\n🧮 Plausibility Group Counts:")
        print("-" * 50)
        for group, cnt in counts.items():
            name = {"vp": "Valid+Plausible", "vi": "Valid+Implausible",
                    "ivp": "Invalid+Plausible", "ivi": "Invalid+Implausible"}.get(group, group)
            print(f"{name:<25}: {cnt}")


# ------------------------------
# ▶️ 主程序入口
# ------------------------------

def main():

    import argparse
    parser = argparse.ArgumentParser(description="Train syllogism validity classifier on augmented data, evaluate on original")
    parser.add_argument(
        "--train_data_paths",
        nargs='+',
        default=[
            "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/unvalidity_plausibility.json",
            "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/unvalidity_unplausibility.json",
            "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/validity_plausibility.json",
            "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/validity_unplausibility.json"
        ],
        help="Paths to augmented training JSON files"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="/home/luorongchuan/workspace_134/Semeval2026/data/merged_data/merged_output.json",
        help="Path to original test JSON file"
    )
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-mnli")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # 在 main() 中调用
    set_all_seeds(args.seed)
    # 设置随机种子


    # 加载训练数据（增强）
    print("📥 Loading augmented training data...")
    train_data = load_and_process_data(args.train_data_paths)
    # 打乱数据顺序（in-place shuffle）
    random.shuffle(train_data)
    for i, ex in enumerate(train_data, start=1):
        ex["id"] = i

    # 验证
    print("First sample ID:", train_data[0].get("id"))
    print("Last sample ID:", train_data[-1].get("id"))
    print(train_data[:10])
    print("Total samples:", len(train_data))
    print(f"✅ Total training examples: {len(train_data)}")

    # 加载测试数据（原始）
    print("📥 Loading original test data...")
    test_data = load_test_data(args.test_data_path)
    print(f"✅ Test examples: {len(test_data)}")

    # 配置
    cfg = NLIConfig(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed
    )

    # 训练 + 测试
    results = train_model(train_data, test_data, cfg)
    print("🎉 Training and evaluation completed!")


if __name__ == "__main__":
    main()