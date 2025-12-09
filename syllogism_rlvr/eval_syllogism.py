# eval_syllogism.py
"""
标准三段论评估模块（用于 SemEval 2026 Task）
提供 compute_content_effects 函数，计算 accuracy、content effect、smoothed score 等指标。
"""

import numpy as np
import math
from typing import List, Dict, Any

def _to_bin_bool(x) -> int:
    """将字符串或布尔值转为二值标签（1=valid, 0=invalid）"""
    if isinstance(x, str):
        x = x.strip().lower()
        if "valid" in x:
            return 1
        elif "invalid" in x:
            return 0
        else:
            raise ValueError(f"Unrecognized label string: '{x}'")
    return int(bool(x))


def compute_content_effects(preds: List[int], labels: List[int], plaus: List[int]) -> Dict[str, Any]:
    """
    计算内容效应指标，与官方评估脚本完全一致。
    
    Args:
        preds: 模型预测 [0/1] 列表
        labels: 真实有效性标签 [0/1]
        plaus: 内容可信度标签 [0=implausible, 1=plausible]
    
    Returns:
        包含 accuracy, content effects, smoothed score 等的字典
    """
    preds = np.asarray(preds)
    labels = np.asarray(labels)
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

    # 平滑得分公式：accuracy / (1 + ln(1 + content_effect))
    if tot_content_effect < 0:
        combined_smooth_score = 0.0
    else:
        log_penalty = math.log(1 + tot_content_effect)
        combined_smooth_score = overall_acc / (1 + log_penalty)

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