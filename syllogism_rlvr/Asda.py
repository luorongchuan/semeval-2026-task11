# test_sft_output.py （修正版）
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_path = "./sft_syllogism_full"
test_json_path = "./data/test.json"  # ← 改成你的路径

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=False,
)
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# === 加载测试数据 ===
with open(test_json_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

# === 构造 prompt：必须和训练时完全一致！===
def build_prompt(syllogism: str) -> str:
    """
    使用和训练时相同的逻辑构造 user message，
    然后通过 apply_chat_template 生成标准 prompt。
    """
    # 1. 分割三段论（和训练时一样）
    from utils import safe_split_syllogism
    p1, p2, c = safe_split_syllogism(syllogism)
    if not all([p1, p2, c]):
        # fallback: 直接用原句（但可能效果差）
        user_msg = f"{syllogism}\n\nIs this syllogism logically valid? Answer only 'valid' or 'invalid'."
    else:
        user_msg = (
            f"Premise 1: {p1}\n"
            f"Premise 2: {p2}\n"
            f"Conclusion: {c}\n\n"
            "Is this syllogism logically valid? Answer only 'valid' or 'invalid'."
        )
    
    # 2. 使用官方 template 生成 prompt（关键！）
    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # ← 生成 [INST] ... [/INST] 格式，不包含 assistant 回复
    )
    return prompt

# === 推理 & 评估 ===
correct = 0
total = len(test_data)

for i, item in enumerate(test_data):
    syllogism = item["syllogism"]
    ground_truth = "valid" if item["validity"] else "invalid"

    prompt = build_prompt(syllogism)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            min_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 截断到第一个 EOS
    generated_ids = outputs[0][input_len:]
    eos_idx = (generated_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_idx) > 0:
        generated_ids = generated_ids[:eos_idx[0] + 1]

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
    prediction = decoded.split()[0] if decoded.split() else ""

    is_correct = (prediction == ground_truth)
    if is_correct:
        correct += 1

    if i < 3 or not is_correct:
        print(f"\n--- Example {i+1} ---")
        print(f"Syllogism: {syllogism}")
        print(f"Prompt preview: {repr(prompt[:100])}...")
        print(f"Expected: {ground_truth} | Predicted: '{prediction}' ({'✅' if is_correct else '❌'})")

accuracy = correct / total
print(f"\n🎯 Final Accuracy: {correct}/{total} = {accuracy:.2%}")