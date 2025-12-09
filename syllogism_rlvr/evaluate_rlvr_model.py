# evaluate_rlvr_model.py
"""
加载 RLVR 微调后的生成式模型（从 PPO checkpoint 提取 base model + LoRA），
对测试集进行推理，并调用标准评估。
"""
import os

# HF 镜像加速（国内必备）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead
from eval_syllogism import compute_content_effects, _to_bin_bool
from peft import PeftModel

# ==============================
# 🔧 配置
# ==============================

TRAINED_MODEL_PATH = "./rlvr_syllogism"      # PPO trainer 保存的完整路径（含 v_head）
BASE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
TEST_FILE = "./data/test.json"

MAX_NEW_TOKENS = 16
TEMPERATURE = 0.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def load_trained_model():
    print(f"Loading PPO-trained model from: {TRAINED_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    
    # 1. 加载 base model（不带 LoRA）
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=False,
    )
    
    # 2. 使用 PeftModel 加载 LoRA adapter（关键！）
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        TRAINED_MODEL_PATH,  # 注意：这里指向的是包含 adapter_model.safetensors 的目录
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    )
    
    # 3. 合并 LoRA 权重（用于推理）
    model = model_with_lora.merge_and_unload()
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("✅ Model loaded successfully")
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Total params:", sum(p.numel() for p in model.parameters()))
    return model, tokenizer


def format_prompt(syllogism: str) -> str:
    """
    ⚠️ 必须与训练时 prepare_ppo_dataset 中的 prompt 完全一致！
    根据你之前的打印，训练 prompt 形如：
        <s>[INST] Premise 1: ...\nPremise 2: ...\nConclusion: ...\n\nIs this syllogism logically valid? Answer only 'valid' or 'invalid'.[/INST]
    """
    # 注意：syllogism 字段本身应包含 Premise/Conclusion 结构
    return (
        f"<s>[INST] {syllogism}\n\n"
        f"Is this syllogism logically valid? Answer only 'valid' or 'invalid'.[/INST]"
    )


def extract_prediction(generated_text: str, input_prompt: str) -> str:
    # 移除 prompt（如果存在）
    if input_prompt in generated_text:
        response = generated_text.split(input_prompt)[-1].strip()
    else:
        response = generated_text.strip()
    
    # 使用正则提取第一个 valid/invalid
    match = re.search(r"\b(valid|invalid)\b", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        print(f"⚠️ Failed to parse prediction from: '{generated_text}' → defaulting to 'invalid'")
        return "invalid"


@torch.no_grad()
def run_inference(model, tokenizer, test_data):
    predictions = []
    for i, ex in enumerate(test_data):
        prompt = format_prompt(ex["syllogism"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred_str = extract_prediction(generated, prompt)
        predictions.append(pred_str)
        
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(test_data)}] Sample processed")
    return predictions


def main():
    # 1. 加载测试数据
    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"✅ Loaded {len(test_data)} test examples")

    # 2. 加载模型
    model, tokenizer = load_trained_model()

    # 3. 推理
    print("🧠 Running inference...")
    raw_preds = run_inference(model, tokenizer, test_data)

    # 4. 转换为二值标签
    y_pred = [_to_bin_bool(p) for p in raw_preds]
    y_true = [_to_bin_bool(ex["validity"]) for ex in test_data]
    plaus = [_to_bin_bool(ex["plausibility"]) for ex in test_data]

    # 5. 评估
    print("📊 Evaluating...")
    metrics = compute_content_effects(y_pred, y_true, plaus)

    # 6. 输出结果
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Accuracy               : {metrics['accuracy']:.4f}%")
    print(f"Total Content Effect   : {metrics['tot_content_effect']:.4f}")
    print(f"Combined Smooth Score  : {metrics['combined_smooth_score']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()