import os
import json
import random
import re
from typing import List, Tuple

# ========== 外网镜像（可选，不影响本任务） ==========
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# -----------------------------
# 工具函数：解析三段论 & 标签转换
# -----------------------------

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
            j = (i+1 + min(cand2)) if cand2 else -1
        return i, j

    i, j = _first_two_delims(text)
    p1 = text[:i].strip() if i != -1 else text
    p2 = text[i+1:j].strip() if j != -1 else ""
    c  = text[j+1:].strip() if j != -1 else ""
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
    if isinstance(x, (int,)):
        return 1 if int(x) == 1 else 0
    return None

# -----------------------------
# 数据加载与处理
# -----------------------------

def load_and_process_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
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

def load_irrelevant_premises(irrelevant_file: str) -> List[str]:
    with open(irrelevant_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        return data
    else:
        raise ValueError(f"Expected a list of strings in {irrelevant_file}")

def augment_with_irrelevant_premises(original_data: List[dict], irrelevant_premises: List[str]) -> List[dict]:
    augmented = []
    for ex in original_data:
        p1 = ex['premise1']
        p2 = ex['premise2']
        conclusion = ex['conclusion']
        validity = ex['labels']
        plausibility = ex['plausibility']

        # 随机选两条无关前提
        ir1, ir2 = random.sample(irrelevant_premises, k=2)

        # 原始有效前提是 p1, p2（索引 0,1）
        premises = [p1, p2, ir1, ir2]
        indices = [0, 1, 2, 3]
        shuffled_indices = indices.copy()
        random.shuffle(shuffled_indices)

        shuffled_premises = [premises[i] for i in shuffled_indices]

        # 记录哪些位置是原始有效前提（即原 index 0 或 1）
        validity_positions = [
            new_idx for new_idx, old_idx in enumerate(shuffled_indices)
            if old_idx in [0, 1]
        ]
        validity_positions.sort()  # 排序便于阅读（非必需）

        new_ex = {
            'premise1': shuffled_premises[0],
            'premise2': shuffled_premises[1],
            'premise3': shuffled_premises[2],
            'premise4': shuffled_premises[3],
            'conclusion': conclusion,
            'validity': validity,
            'plausibility': plausibility,
            # 'validity_premise': validity_positions  # 如 [0, 2]
            'validity_premise': []  # 无效数据没有相关前提
        }
        augmented.append(new_ex)
    return augmented

# -----------------------------
# 主程序：生成增强后的训练数据并保存
# -----------------------------

if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)

    # 训练数据路径（4类）
    train_files = [
        "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/unvalidity_plausibility.json",
        "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/unvalidity_unplausibility.json",
        # "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/validity_plausibility.json",
        # "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/validity_unplausibility.json"
    ]

    # 无关前提文件
    irrelevant_file = "/home/luorongchuan/workspace_134/Semeval2026/A_work_python/third/wuguanqianti.json"

    print("Loading raw training data...")
    raw_train_data = []
    for fp in train_files:
        raw_train_data.extend(load_and_process_data(fp))

    print("Loading irrelevant premises...")
    irrelevant_premises = load_irrelevant_premises(irrelevant_file)

    print("Augmenting each sample with 2 random irrelevant premises...")
    augmented_data = augment_with_irrelevant_premises(raw_train_data, irrelevant_premises)

    print(f"Total augmented samples: {len(augmented_data)}")

    # 保存结果
    output_path = "/home/luorongchuan/workspace_134/Semeval2026/A_work_python/third/augmented_with_irrelevant_invalidity.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved augmented data to: {output_path}")