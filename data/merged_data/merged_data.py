import json

def merge_json_files(file1, file2, output_file):
    # 读取第一个文件
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    # 读取第二个文件
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # 合并数据
    merged_data = data1 + data2

    # 重新分配 id，从 0 开始连续编号
    for idx, item in enumerate(merged_data):
        item["id"] = str(idx)  # 如果你希望 id 是字符串类型；若需整数，用 idx 即可

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)

    print(f"✅ 合并完成！共 {len(merged_data)} 条数据，已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    file_a = "/home/luorongchuan/workspace_134/Semeval2026/data/pilot data/syllogistic_reasoning_binary_pilot_en.json"
    file_b = "/home/luorongchuan/workspace_134/Semeval2026/data/train_data/train_data.json"
    output = "/home/luorongchuan/workspace_134/Semeval2026/data/merged_output.json"
    merge_json_files(file_a, file_b, output)