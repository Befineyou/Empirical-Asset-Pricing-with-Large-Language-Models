import json

# 读取 JSON 文件
with open('train_prompts.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 写入 JSONL 文件
with open('output.jsonl', 'w', encoding='utf-8') as f:
    # 如果 JSON 文件是一个数组，我们逐项写入
    if isinstance(data, list):
        for item in data:
            f.write(json.dumps(item) + '\n')
    # 如果 JSON 文件是一个对象，我们直接写入
    elif isinstance(data, dict):
        f.write(json.dumps(data) + '\n')
