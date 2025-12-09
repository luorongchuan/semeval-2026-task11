# ppo_data.py
# ppo_data.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from typing import Optional, List
from torch.utils.data import Dataset as TorchDataset  # ⚠️ 注意别名
from transformers import PreTrainedTokenizerBase
from utils import load_syllogism_dataset


class PPODataset(TorchDataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def prepare_ppo_dataset(
    tokenizer: PreTrainedTokenizerBase,
    data_path: str,
    max_length: int = 512,
    split: str = "train",
) -> TorchDataset:
    """
    Prepare a dataset for PPO training as a torch.utils.data.Dataset.
    This avoids field filtering by PPOTrainer (which only happens for datasets.Dataset).
    """
    prompts, labels = load_syllogism_dataset(
        filepath=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode="ppo"
    )

    if len(prompts) == 0:
        raise ValueError(f"No valid examples found in {data_path}.")

    print(f"✅ Loaded {len(prompts)} examples for PPO ({split}).")

    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        return_token_type_ids=False,
        add_special_tokens=False,
    )

    dataset = PPODataset(
        input_ids=tokenized["input_ids"],
        attention_mask=tokenized["attention_mask"],
        labels=labels,
    )

    # Sanity check
    if len(dataset) > 0:
        ex = dataset[0]
        decoded = tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
        print(f"\n🔍 Example prompt (decoded):\n{decoded}\n")
        print(f"Label: {'valid' if ex['labels'] == 1 else 'invalid'}\n")

    return dataset


if __name__ == "__main__":
    # Simple test script
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    tokenizer.pad_token = tokenizer.eos_token

    data_path = "/home/luorongchuan/workspace_134/Semeval2026/A_work_python/syllogism_rlvr/data/train.json"
    dataset_list = prepare_ppo_dataset(tokenizer, data_path, max_length=256)
    print(f"Dataset size: {len(dataset_list)}")
    print(f"First sample keys: {list(dataset_list[0].keys())}")
    print(f"First label: {dataset_list[0]['labels']}")