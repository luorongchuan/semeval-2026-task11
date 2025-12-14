import os
from transformers import AutoConfig, DebertaV2ForSequenceClassification, DebertaV2Tokenizer
from safetensors.torch import load_file
import torch

MODEL_PATH = "/home/luorongchuan/workspace_134/Semeval2026/Q1/out"

# 1. 加载 tokenizer（显式指定类）
print("Loading tokenizer with DebertaV2Tokenizer...")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)
print("✅ Tokenizer loaded.")

# 2. 加载 config
print("Loading config...")
config = AutoConfig.from_pretrained(MODEL_PATH)
config.num_labels = 2  # 必须设置，否则默认是 2 也可能不对
print("✅ Config loaded.")

# 3. 手动创建模型（不触发权重加载）
print("Creating model from config...")
model = DebertaV2ForSequenceClassification(config)
print("✅ Model structure created.")

# 4. 手动加载 safetensors 权重
print("Loading weights from model.safetensors...")
state_dict = load_file(os.path.join(MODEL_PATH, "model.safetensors"))
missing, unexpected = model.load_state_dict(state_dict, strict=False)
if missing:
    print("⚠️ Missing keys:", missing)
if unexpected:
    print("⚠️ Unexpected keys:", unexpected)
print("✅ Weights loaded.")

# 5. 测试推理
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()
text = "This is a test sentence."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
with torch.no_grad():
    logits = model(**inputs).logits
    pred = logits.argmax(dim=-1).item()
print(f"✅ Inference works! Prediction: {pred}")