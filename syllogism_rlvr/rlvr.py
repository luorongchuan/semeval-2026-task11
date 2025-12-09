import os

# HF 镜像加速（国内必备）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["WANDB_MODE"] = "disabled"  # 完全禁用 W&B
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from ppo_data import prepare_ppo_dataset
from utils import compute_verifiable_reward
import warnings
warnings.filterwarnings("ignore")

# ==============================
# 配置
# ==============================
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
sft_model_path = "./sft_syllogism_full"
data_path = "./data/ppo_train.json"
output_dir = "./rlvr_syllogism"

ppo_config = PPOConfig(
    batch_size=128,
    mini_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    log_with=None,
    steps=50,
    optimize_cuda_cache=True,
    seed=42,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],
)

# ==============================
# Tokenizer & Reward Token IDs
# ==============================
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 🔥 预计算 valid/invalid 的 token ID（关键提速点！）
valid_id = tokenizer("valid", add_special_tokens=False)["input_ids"][0]
invalid_id = tokenizer("invalid", add_special_tokens=False)["input_ids"][0]
print(f"✅ Valid token ID: {valid_id}, Invalid token ID: {invalid_id}")

# ==============================
# 模型加载
# ==============================
print("Loading SFT model in bfloat16...")
base_model = AutoModelForCausalLM.from_pretrained(
    sft_model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    trust_remote_code=False,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model,
    peft_config=lora_config,
)

print("Loading reference model...")
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model,
    peft_config=lora_config,
)
for param in ref_model.parameters():
    param.requires_grad = False

# ==============================
# 数据集
# ==============================
print("Preparing PPO dataset...")
dataset = prepare_ppo_dataset(
    tokenizer=tokenizer,
    data_path=data_path,
    max_length=256,
    split="train"
)

def collator(data):
    return {
        key: [d[key] for d in data]
        for key in ["input_ids", "attention_mask", "labels"]
    }

# ==============================
# PPO Trainer
# ==============================
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

# 🔥 固定生成长度（足够输出 "valid"/"invalid"）
generation_kwargs = {
    "min_new_tokens": 4,
    "max_new_tokens": 8,      # ⚡ 小而固定
    "top_k": 20,
    "top_p": 0.92,
    "do_sample": True,
    "temperature": 0.5,
    "pad_token_id": tokenizer.eos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "use_cache": True,
}

# ==============================
# 高效 Reward 函数（无 decode！）
# ==============================
def compute_verifiable_reward_from_ids(response_tensors, true_labels, valid_id, invalid_id):
    rewards = []
    for resp_tensor, label in zip(response_tensors, true_labels):
        if isinstance(label, str):
            label = int(label.strip())
        
        # 只看第一个生成的 token（模型应只输出一个词）
        first_token = resp_tensor[0].item()

        if first_token == valid_id:
            pred = 1
        elif first_token == invalid_id:
            pred = 0
        else:
            rewards.append(0.0)
            continue

        rewards.append(1.0 if pred == label else 0.0)
    return rewards

# ==============================
# 训练循环
# ==============================
print("🚀 Starting PPO training (optimized for speed)...")
for epoch, batch in enumerate(ppo_trainer.dataloader):
    if epoch >= ppo_config.steps:
        break

    query_tensors = [
        torch.tensor(q).to(ppo_trainer.accelerator.device)
        for q in batch["input_ids"]
    ]

    # 生成 responses（batched via loop, but fast due to fixed length）
    response_tensors = []
    for query in query_tensors:
        response = ppo_trainer.generate(query, **generation_kwargs)
        # 取最后 max_new_tokens 个 token（实际就是生成部分）
        new_response = response.squeeze()[-generation_kwargs["max_new_tokens"]:]
        response_tensors.append(new_response)

    # 🔥 高速 reward 计算（无 decode！）
    rewards_list = compute_verifiable_reward_from_ids(
        response_tensors, batch["labels"], valid_id, invalid_id
    )
    rewards = [
        torch.tensor(float(r), dtype=torch.float32).to(ppo_trainer.accelerator.device)
        for r in rewards_list
    ]

    # PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    # Logging
    def to_scalar(x):
        return x.item() if hasattr(x, "item") else float(x) if x is not None else float("nan")

    policy_loss = to_scalar(stats.get("ppo/loss/policy", float("nan")))
    value_loss = to_scalar(stats.get("ppo/loss/value", float("nan")))
    reward_mean = to_scalar(stats.get("ppo/mean_scores", float("nan")))

    print(f"✅ Step {epoch} | "
          f"Policy Loss: {policy_loss:.4f} | "
          f"Value Loss: {value_loss:.4f} | "
          f"Reward Mean: {reward_mean:.4f}")

    # Save checkpoint every 50 steps
    if epoch % 50 == 0 and epoch > 0:
        save_path = os.path.join(output_dir, f"checkpoint-{epoch}")
        ppo_trainer.save_pretrained(save_path)
        print(f"💾 Saved checkpoint to {save_path}")

# Final save
ppo_trainer.save_pretrained(output_dir)
print(f"🎉 Training finished. Final model saved to {output_dir}")