import os
import random
import json
import pandas as pd
from typing import List, Dict, Union
from dataclasses import dataclass
from itertools import product
from datasets import Dataset




@dataclass
class Config:
    train_epochs: int = 100
    val_epochs: int = 1
    test_data_type: str = "believable"
    schemes_dir: str = "Augmentation_data/soft-syllogistic-reasoners-main/data/schemes"
    vocab_dir: str = "Augmentation_data/soft-syllogistic-reasoners-main/data/vocabulary"
    test_dir: str = "Augmentation_data/soft-syllogistic-reasoners-main/data/test"
    valid_test_data: List[str] = (
        "believable",
        "unbelievable",
        "2_premises",
        "3_premises",
        "4_premises",
    )


class SyllogismProcessor:
    def __init__(self, config: Config):
        self.config = config
        self._create_directories()
        self._initialize_constants()
        self._validate_config()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [self.config.schemes_dir, self.config.vocab_dir, self.config.test_dir]:
            os.makedirs(directory, exist_ok=True)

    def _initialize_constants(self) -> None:
        """Initialize constant values used throughout the process."""
        self.quantifiers = {
            "len_1": {
                "A": "All X are Y",
                "I": "Some X are Y",
                "O": "Some X are not Y",
                "E": "No X are Y"
            },
            "len_2": {
                "A": "All X are Z. All Z are Y",
                "I": "Some X are Y",
                "O": "Some X are not Y",
                "E": "No X are Y"
            },
            "len_3": {
                "A": "All X are Z. All Z are W. All W are Y",
                "I": "Some X are Y",
                "O": "Some X are not Y",
                "E": "No X are Y"
            }
        }

        self.figures = {
            "1": [("A", "B"), ("B", "C")],
            "2": [("B", "A"), ("C", "B")],
            "3": [("A", "B"), ("C", "B")],
            "4": [("B", "A"), ("B", "C")],
        }

        self.valid_syllogisms = {
            "AA1" : ["All A are C", "some A are C", "some C are A"],
            "IA1" : ["Some A are C", "some C are A"],
            "EA1" : ["Some C are not A"],
            "EI1" : ["Some C are not A"],
            "AE1" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "IE1" : ["Some A are not C"],
            "EA2" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "AI2" : ["Some A are C", "some C are A"],
            "AA2" : ["All C are A", "some A are C", "some C are A"],
            "EI2" : ["Some C are not A"],
            "IE2" : ["Some A are not C"],
            "AE2" : ["Some A are not C"],
            "EA3" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "AE3" : ["No A are C", "no C are A", "some A are not C", "some C are not A"],
            "AO3" : ["Some C are not A"],
            "OA3" : ["Some A are not C"],
            "IE3" : ["Some A are not C"],
            "EI3" : ["Some C are not A"],
            "AI4" : ["Some A are C", "some C are A"], 
            "IA4" : ["Some A are C", "some C are A"],
            "AO4" : ["Some A are not C"],
            "OA4" : ["Some C are not A"],
            "AA4" : ["Some A are C", "some C are A"],
            "IE4" : ["Some A are not C"],
            "EI4" : ["Some C are not A"],
            "AE4" : ["Some A are not C"],
            "EA4" : ["Some C are not A"],
        }

        self.options = [
            "All A are C.",
            "Some A are C.",
            "No A are C.",
            "Some A are not C.",
            "All C are A.",
            "Some C are A.",
            "No C are A.",
            "Some C are not A."
            # "Nothing follows."
        ]
        self.plausibility_syllogisms = {
                    "AA3" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "AI1" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "AI3" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "AO1" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "AO2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "IA2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "IA3" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "II1" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "II2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "II3" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "II4" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "IO1" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "IO2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "IO3" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "IO4" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "OA1" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "OA2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "OI1" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "OI2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "OI3" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "OI4" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "OO1" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "OO2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "OO3" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "OO4" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                    "OE1" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "OE2" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "OE3" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "OE4" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "EO1" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "EO2" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "EO3" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "EO4" : ["No A are C","No C are A", "Some A are not C", "Some C are not A"],
                    "EE1" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "EE2" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "EE3" : ["All A are C", "some A are C", "some C are A","Some C are not A"],
                    "EE4" : ["All C are A", "some C are A", "some A are C","Some A are not C"],
                }


    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.test_data_type not in self.config.valid_test_data:
            raise ValueError(
                f"Invalid test_data_type: {self.config.test_data_type}. "
                f"Must be one of: {', '.join(self.config.valid_test_data)}"
            )

    def generate_schemes(self, gibberish: bool = False, len_chain: int = 1) -> None:
        """Generate syllogism schemes and save them to files."""
        structures = []
        answers = []
        names = []

        permutations = list(product(self.quantifiers["len_1"].keys(), repeat=2))
        
        for quantifier in permutations:
            if not gibberish:
                for figure in self.figures.keys():
                    major_name, minor_name = quantifier
                    major_x, major_y = self.figures[figure][0]
                    minor_x, minor_y = self.figures[figure][1]
                    name = f"{major_name}{minor_name}{figure}"

                    # Generate premises of standard syllogisms
                    premise_1 = self.quantifiers["len_1"][major_name].replace("X", major_x).replace("Y", major_y)
                    premise_2 = self.quantifiers["len_1"][minor_name].replace("X", minor_x).replace("Y", minor_y)

                    structure = f"Premise 1: {premise_1} / Premise 2: {premise_2}"
                    answer = " | ".join(self.valid_syllogisms.get(name, ["Nothing follows"]))

                    structures.append(structure)
                    answers.append(answer)
                    names.append(name)

            elif "A" in quantifier:
                for figure in self.figures.keys():
                    major_name, minor_name = quantifier
                    major_x, major_y = self.figures[figure][0]
                    minor_x, minor_y = self.figures[figure][1]
                    name = f"{major_name}{minor_name}{figure}"

                    # Generate premises using gibberish and based on chain length
                    premise_1 = self.quantifiers[f"len_{len_chain}"][major_name].replace("X", major_x).replace("Y", major_y)
                    premise_2 = self.quantifiers[f"len_{len_chain}"][minor_name].replace("X", minor_x).replace("Y", minor_y)

                    structure = []
                    premise_a = premise_1 if len(premise_1) >= len(premise_2) else premise_2
                    pos_premise_a = 1 if premise_a == premise_1 else 2

                    if pos_premise_a == 2:
                        premise_1 = premise_1.replace("Z. All Z are W. All W are ", "").replace("Z. All Z are ", "")
                        structure.append(f"Premise 1: {premise_1}")

                    premise_a = premise_a.split(".")
                    for n, a in enumerate(premise_a):
                        structure.append(f"Premise {n+pos_premise_a}: {a}")

                    if pos_premise_a == 1:
                        premise_2 = premise_2.replace("Z. All Z are W. All W are ", "").replace("Z. All Z are ", "")
                        structure.append(f"Premise {len_chain+1}: {premise_2}")

                    structure = " / ".join(structure).replace("  ", " ")
                    answer = " | ".join(self.valid_syllogisms.get(name, ["Nothing follows"]))

                    structures.append(structure)
                    answers.append(answer)
                    names.append(name)

        # Save schemes to file
        filename = (f"schemes_syllogisms_{len_chain+1}.txt" if gibberish 
                   else "schemes_syllogisms.txt")
        path = os.path.join(self.config.schemes_dir, filename)
        
        with open(path, "w") as f:
            for name, structure, answer in zip(names, structures, answers):
                f.write(f"<ID> {name}\n<STRUCTURE> {structure}\n<CONCLUSION> {answer}\n####\n")

    def _schemes_exist(self) -> bool:
        """Check if all necessary scheme files exist."""
        basic_scheme = os.path.join(self.config.schemes_dir, "schemes_syllogisms.txt")
        if not os.path.exists(basic_scheme):
            return False

        if "premises" in self.config.test_data_type:
            chain_length = int(self.config.test_data_type[0])
            chain_scheme = os.path.join(self.config.schemes_dir, f"schemes_syllogisms_{chain_length}.txt")
            return os.path.exists(chain_scheme)

        return True

    def ensure_schemes_exist(self) -> None:
        """Generate schemes if they don't exist."""
        if not self._schemes_exist():
            print("Generating schemes...")
            # Generate standard schemes
            self.generate_schemes(gibberish=False)
            # Generate gibberish schemes if needed
            if "premises" in self.config.test_data_type:
                chain_length = int(self.config.test_data_type[0]) - 1
                self.generate_schemes(gibberish=True, len_chain=chain_length)
        else:
            print("Schemes already exist. Skipping generation.")


class SyllogismDataset(SyllogismProcessor):
    def __init__(self, config: Config):
        super().__init__(config)
        self.ensure_schemes_exist()
        self._load_vocabulary()

    def _load_vocabulary(self) -> None:
        """Load the appropriate vocabulary based on test data type."""
        vocab_path = os.path.join(
        self.config.vocab_dir,
        f"syllowords_{self.config.test_data_type}.json"
    )
        

        with open(vocab_path) as f:
            self.syllowords = json.load(f)

    def _load_schemes(self, split: str) -> List[str]:
    # """Load syllogism schemes based on split and test data type."""
    # 根据 test_data_type 和数据划分(split)决定要加载哪个 schemes 文件
        if "premises" in self.config.test_data_type and split == "test":
            # 例如 test_data_type = "3_premises" -> 用 schemes_syllogisms_3.txt
            filename = f"schemes_syllogisms_{self.config.test_data_type[0]}.txt"
        else:
            # 普通情况都用基础的 schemes_syllogisms.txt
            filename = "schemes_syllogisms.txt"

        schemes_path = os.path.join(self.config.schemes_dir, filename)

        with open(schemes_path) as f:
            # 按照 "\n####\n" 分块，每个块对应一个三段论形式
            # 最后一个 split 之后通常是空字符串，所以用 [:-1] 去掉
            return f.read().split("\n####\n")[:-1]



    # 有效合理
    def _process_structure(self, structure: str, problem_type: str) -> List[tuple]:
        # """处理单个结构，生成多个样本，并根据替换词判断有效性和合理性"""
        results = []

        # 根据 problem_type 获取对应的替换词集合
        to_substitute = self.syllowords[problem_type]  # 获取与问题类型相关的词汇

        # 提取并格式化结论部分
        conclusion = structure.split("<CONCLUSION> ")[1].replace("|", "or").strip()  # 不加句号，分割时方便处理
        conclusions = conclusion.split(" or ")  # 拆分为多个结论

        # 如果结论是 "Nothing follows"，跳过该数据
        if "nothing follows" in [c.strip().lower() for c in conclusions]:
            return results  # 直接跳过该结构，返回空的结果

        # 遍历每组替换词
        for _ in range(len(to_substitute)):
            # 提取文本的前提部分
            txt = structure.split("\n<STRUCTURE> ")[1].split("\n<CONCLUSION>")[0].replace(" / ", ".\n") + "."
            txt = f"Syllogism:{txt} "
            raw = txt

            # 替换前提中的 A/B/C
            noun_a, noun_b, noun_c = to_substitute[_]
            for old, new in [("A", noun_a), ("B", noun_b), ("C", noun_c)]:
                txt = txt.replace(f"{old} ", f"{new} ").replace(f"{old}.", f"{new}.")
            
            # 遍历每个结论，生成一个样本
            for c in conclusions:
                ans_replaced = c.strip() + "."  # 为结论加上句号

                # 替换结论中的 A/B/C
                for old, new in [("A", noun_a), ("B", noun_b), ("C", noun_c)]:
                    ans_replaced = ans_replaced.replace(f"{old} ", f"{new} ").replace(f"{old}.", f"{new}.")

                is_valid = True  # 默认是有效的，因为前提和结论都是从同一个替换词表中取词的

                # 根据 test_data_type 判断合理性
                plausibility = None
                if self.config.test_data_type == "believable":
                    plausibility = True  # 合理
                else:
                    plausibility = False  # 不合理

                # 生成有效性和合理性的判断
                results.append((raw, txt, ans_replaced, c.strip(), is_valid, plausibility))

        return results



    def create_data_df(self, split: str = "train") -> pd.DataFrame:
        """Create a DataFrame with syllogism data."""
        syllo_structures = self._load_schemes(split)
        
        # Initialize lists for problem types, texts, structures, answers, and other data
        problem_types = []
        texts, structures, answers, ans_types = [], [], [], []  # Changed 'ans_type' to 'ans_types' to avoid conflict with the list
        validities, plausibilities = [], []
        
        for structure in syllo_structures:
            problem_type = structure.split("\n<STRUCTURE>")[0].replace("<ID> ", "")
            processed_examples = self._process_structure(structure, problem_type)
            
            # Populate the lists with processed data
            for raw, txt, ans_replaced, ans_type, val, plaus in processed_examples:
                problem_types.append(problem_type)
                texts.append(txt)
                structures.append(raw)
                answers.append(ans_replaced)
                ans_types.append(ans_type)  # Correctly append 'ans_type' to 'ans_types' list
                validities.append(bool(val))
                plausibilities.append(bool(plaus))

        # Create and return the DataFrame with the collected data
        return pd.DataFrame({
            "problem_type": problem_types,
            "text": texts,
            "structures": structures,
            "answers": answers,
            "ans_type": ans_types,  # Ensure consistent naming with 'ans_types' (plural)
            "validity": validities,
            "plausibility": plausibilities,
        })


    def substitute_words(self, df: pd.DataFrame, split: str = "train"):
        # """Substitute words in the syllogism data with gibberish to create content-biased variants."""
        gibberish_path = os.path.join(self.config.vocab_dir, "gibberish.json")
        with open(gibberish_path) as f:
            words = json.load(f)[split]
        
        # # Load believable vocabulary for plausible conclusion substitution
        # unbelievable_vocab_path = os.path.join(self.config.vocab_dir, "syllowords_unbelievable.json")
        # with open(unbelievable_vocab_path) as f:
        #     unbelievable_vocab = json.load(f)  # dict like {"AA1": [["dog", "animal", "mammal"], ...], ...}
        substituted_data = []

        for _, row in df.iterrows():
            original_text = row["text"]  # Full original syllogism string (e.g., "All A are B. All B are C.")
            structure = row["structures"]  # Template ending with "Answer: ..."
            answers = row["answers"]  # Short answer (e.g., "Yes")
            answer_full = row["ans_type"]  # Full conclusion (e.g., "All A are C.")
            validity = bool(row["validity"])
            plausibility = bool(row["plausibility"])
            problem_type = row["problem_type"]


            # --- Case 1: Original (no substitution) ---
            substituted_data.append((original_text, answers, validity, plausibility))

            # Prepare token mapping for this example
            mapping_tokens = random.sample(words, 5)
            token_map = {
                "A": mapping_tokens[0],
                "B": mapping_tokens[1],
                "C": mapping_tokens[2],
                "Z": mapping_tokens[3],
                "W": mapping_tokens[4],
            }

            # Helper function to safely replace tokens (handles spaces and periods)
            def replace_tokens(text: str, mapping: dict) -> str:
                result = text
                for old, new in mapping.items():
                    result = result.replace(f"{old} ", f"{new} ")
                    result = result.replace(f"{old}.", f"{new}.")
                return result

            # Extract premise part (everything before "Answer: ")
          
            # if (problem_type in unbelievable_vocab 
            # and unbelievable_vocab[problem_type] 
            # and random.random() < 0.3):  # ← 关键：30% 替换概率
            
            #     noun_a, noun_b, noun_c = random.choice(unbelievable_vocab[problem_type])
            #     premise_part = structure.split("Answer:")[0].strip()
                
            #     for ph, word in [("A", noun_a), ("B", noun_b), ("C", noun_c)]:
            #         premise_part = premise_part.replace(f"{ph} ", f"{word} ")
            #         premise_part = premise_part.replace(f"{ph}.", f"{word}.")
                
                # substituted_data.append((premise_part, answers, False, plausibility))

        return substituted_data





    def _generate_split(self, split: str) -> Dataset:
        """Generate dataset for a specific split."""
        if split == "train":
            # 生成训练数据
            base_df = self.create_data_df(split="train")
            all_samples = []
            for _ in range(self.config.train_epochs):
                all_samples.extend(self.substitute_words(base_df, split="train"))

            # 确保返回的每一行都包含 4 个字段
            texts = [t for t, _, _, _ in all_samples]
            answers = [a for _, a, _, _ in all_samples]
            validities = [v for _, _, v, _ in all_samples]
            plausibilities = [p for _, _, _, p in all_samples]

            return Dataset.from_dict({
                "text": texts,
                "answer": answers,
                "validity": validities,
                "plausibility": plausibilities,
            })

        elif split == "val":
            # 生成验证数据
            df = self.create_data_df(split="val")
            val_substituted = []
            for _ in range(self.config.val_epochs):
                val_substituted.extend(self.substitute_words(df, split="val"))

            # 这里做解包操作，确保有 4 个字段
            return Dataset.from_dict({
                "text": [text for text, _, _, _ in val_substituted],
                "answer": [ans for _, ans, _, _ in val_substituted],
                "validity": [validity for _, _, validity, _ in val_substituted],
                "plausibility": [plausibility for _, _, _, plausibility in val_substituted],
            })

        else:  # test split
            return self._generate_test_split()


    def _generate_test_split(self) -> Dataset:
        test_path = self._get_test_path()

        # 如果文件不存在，直接创建
        if not os.path.exists(test_path):
            self._create_test_file(test_path)
        else:
            # 检查一下旧文件里有没有 validity/plausibility
            tmp = pd.read_json(test_path, lines=True)
            if "validity" not in tmp.columns or "plausibility" not in tmp.columns:
                # 旧格式文件，重新创建覆盖
                self._create_test_file(test_path)

        test_data = pd.read_json(test_path, lines=True)
        return Dataset.from_pandas(test_data)


    def _get_test_path(self) -> str:
        """Get the appropriate test file path."""
        return os.path.join(self.config.test_dir, f"syllogisms_{self.config.test_data_type}.jsonl")

    def _create_test_file(self, path: str) -> None:
        """Create test file if it doesn't exist."""
        test_data = self.create_data_df(split="test")
        
        if "premises" in self.config.test_data_type:
            examples = []
            for _ in range(10):
                examples.extend(self.substitute_words(test_data, split="test"))
                
            types = [test_data["type"].iloc[i] for i in range(0, len(test_data), 10)] * 10
            term_orders = [test_data["term_order"].iloc[i] for i in range(0, len(test_data), 10)] * 10
            validities = [self._is_valid(t) for t in types]
            plausibilities = [self._plausibility_flag() for _ in types]
            test_data = pd.DataFrame({
                "text": [ex[0] for ex in examples],
                "answer": [ex[1] for ex in examples],
                "type": types,
                "term_order": term_orders,
                "validity": validities,
                "plausibility": plausibilities,
            })
        else:
            inputs = [txt.split("Answer: ")[0] + "Answer: " for txt in test_data['text']]
            answers = [txt.split("Answer: ")[1] for txt in test_data['text']]
            test_data["text"] = inputs
            test_data["answer"] = answers
            test_data["validity"] = test_data["type"].map(lambda t: self._is_valid(t))
            test_data["plausibility"] = self._plausibility_flag()
            test_data = test_data[['text', 'answer', 'type', 'term_order', 'validity', 'plausibility']]
            
        test_data.to_json(path, orient='records', lines=True)

    def get_dataset(self) -> Dict[str, Dataset]:
        """Generate complete dataset with all splits."""
        return {
            "train": self._generate_split("train")
        }


def get_dataset(
    train_epochs: int = 100,
    val_epochs: int = 1,
    test_data_type: str = "believable"
) -> Dict[str, Dataset]:
    """Convenience function to create and return a dataset."""
    config = Config(
        train_epochs=train_epochs,
        val_epochs=val_epochs,
        test_data_type=test_data_type
    )
    dataset = SyllogismDataset(config)
    return dataset.get_dataset()


import uuid
import pandas as pd

# 假设 get_dataset 函数已经定义并且能获取到数据
# def get_dataset(...)  # 此处略去函数定义

if __name__ == "__main__":

    dataset = get_dataset(
        train_epochs=1,
        val_epochs=1,
        test_data_type="believable"
    )

    train_df = dataset["train"].to_pandas()
    # 确认数据是否正确加载
    print(train_df.head())  # 查看前几行数据
    print(f"Train DataFrame shape: {train_df.shape}")  # 打印 DataFrame 的形状

    # 处理数据
    converted_data = []
    for index, item in train_df.iterrows():  # 使用 iterrows() 遍历 DataFrame
        # 去掉 "Premise 1" 和 "Premise 2" 以及 "Syllogism:"
        text = item["text"].replace("Syllogism:", "").replace("Premise 1:", "").replace("\nPremise 2:", "").replace("Answer:", "").strip()
        # 组合前提和结论
        syllogism = text + " " + item["answer"]
        
        # 使用处理的顺序索引作为 ID
        unique_id = str(index)  # 使用 index 作为唯一的 ID
        
        converted_data.append({
            "id": unique_id,  # 生成唯一的 ID
            "syllogism": syllogism.strip(),
            "validity": item["validity"],
            "plausibility": item["plausibility"]
        })

    # 转换数据为 DataFrame 并保存为 JSON
    output_file = "/home/luorongchuan/workspace_134/Semeval2026/Augmentation_data/soft-syllogistic-reasoners-main/scripts/dataset3/validity_plausibility.json"  # 可以指定完整路径
    # 将生成的数据存储为 JSON 格式
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=4)

    # 确认文件写入
    print(f"Train dataset generated successfully! File saved to: {output_file}")
    print(f"Train samples: {len(converted_data)}")
    print("\nTRAINING SAMPLE:")
    print(converted_data[0])  # 打印转换后的数据样本
