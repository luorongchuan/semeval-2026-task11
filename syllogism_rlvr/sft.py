# sft.py - 高性能 SFT 训练脚本（显存充足场景）
import os
import shutil
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# 环境设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "5,4,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from utils import load_syllogism_dataset


def main():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    lora_output_dir = "./sft_syllogism_lora"
    full_model_output_dir = "./sft_syllogism_full"

    # ==============================
    # 🔧 Tokenizer & Model (启用 FlashAttention-2)
    # ==============================
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        # use_flash_attention_2=True,  # ✅ 关键加速：FlashAttention-2
    )
    model.gradient_checkpointing_enable()


    # ==============================
    # 🔧 LoRA 配置
    # ==============================
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        # modules_to_save=["embed_tokens", "lm_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ==============================
    # 📊 数据加载
    # ==============================
    train_dataset = load_syllogism_dataset(
        "/home/luorongchuan/workspace_134/Semeval2026/A_work_python/syllogism_rlvr/data/sft_train.json",
        tokenizer,
        max_length=512,
        mode="sft"
    )
    print(f"✅ Loaded {len(train_dataset)} training examples")
    print(train_dataset[0])
     # 🔍【新增：验证第一条样本】
    print("Sample 0 input:", tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=False))
    print("Sample 0 labels (non -100):", [x for x in train_dataset[0]["labels"] if x != -100])

    # ==============================
    # ⚙️ 训练参数（高性能配置）
    # ==============================
    per_device_batch = 32          # 大 batch size（显存够就拉满）
    grad_acc_steps = 1             # 不再需要梯度累加

    training_args = TrainingArguments(
        output_dir=lora_output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        bf16=True,
        optim="adamw_torch_fused",       # ✅ 更快的 fused AdamW
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        seed=42,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=8,        # 提高数据加载并行度
        dataloader_pin_memory=True,
        # prefetch_factor=4,               # 每个 worker 预取 4 个 batch
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=None,
    )

    trainer.train()

    # ==============================
    # 💾 保存 LoRA 适配器
    # ==============================
    final_lora_dir = os.path.join(lora_output_dir, "final")
    model.save_pretrained(final_lora_dir)
    tokenizer.save_pretrained(final_lora_dir)
    print(f"✅ LoRA adapter saved to {final_lora_dir}")

    # ==============================
    # 🔗 合并并保存完整模型（用于 RLVR）
    # ==============================
    print("🔄 Merging LoRA weights into base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # 合并在 CPU 上更安全
    )

    merged_model = PeftModel.from_pretrained(base_model, final_lora_dir)
    merged_model = merged_model.merge_and_unload()

    merged_model.save_pretrained(full_model_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(full_model_output_dir)

    # 复制 chat template（如果存在）
    template_src = os.path.join(final_lora_dir, "chat_template.jinja")
    template_dst = os.path.join(full_model_output_dir, "chat_template.jinja")
    if os.path.exists(template_src):
        shutil.copy(template_src, template_dst)
        print(f"✅ Copied chat_template.jinja to {full_model_output_dir}")
    else:
        print("⚠️ Warning: chat_template.jinja not found in LoRA dir.")

    print(f"✅ Full merged model saved to {full_model_output_dir}")


if __name__ == "__main__":
    main()