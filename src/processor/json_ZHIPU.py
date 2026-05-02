"""ZHIPU AI官方文档：https://docs.bigmodel.cn/api-reference
调用 glm-4.6v 从表格json中提取原始数据，并输出结构化json
提示语见Prompt.txt"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ZHIPU_API_KEY')
MODEL_NAME = os.getenv('ZHIPU_MODEL_NAME')
API_URL = os.getenv('ZHIPU_API_URL')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

JSON_FILE = os.path.join(project_root, os.getenv('JSON_FILE'))
JSON_FOLDER = os.path.dirname(JSON_FILE)
PROMPT_FILE = os.path.join(project_root, os.getenv('PROMPT_FILE'))
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    PROMPT = f.read()


def call_zhipu_stream(file_path, retry_count=3):
    filename = os.path.basename(file_path)
    print(f"正在处理文件 {filename} ...")

    with open(file_path, 'r', encoding='utf-8') as f:
        table_data = json.load(f)
    if not table_data:
        print(f"跳过空文件 {filename}")
        return

    input_content = json.dumps(table_data, ensure_ascii=False)
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"这是你需要处理的json文件内容：\n{input_content}"}
        ],
        "stream": True,  #开启流式（输出思考过程避免timeout）
        "temperature": 0.1,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(retry_count):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=(10, 120))
            response.raise_for_status()
            full_content = ""
            has_reasoning = False
            #逐行解析SSE数据
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if "reasoning_content" in delta: #处理深度思考内容
                                    if not has_reasoning:
                                        print("[思考中]: ", end='', flush=True)
                                        has_reasoning = True
                                    print(delta["reasoning_content"], end='', flush=True)
                                    continue  # 思考内容不拼接到最终结果
                                content = delta.get("content", "")
                                if content:
                                    full_content += content
                                    print(content, end='', flush=True)
                        except json.JSONDecodeError:
                            continue
            return full_content

        except Exception as e:
            print(f"请求失败 (尝试 {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                time.sleep(5)
            else:
                raise e

def parse_tables(response_text): #清洗并解析JSON
    if not response_text:
        return []
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    try:
        data = json.loads(text.strip())
        return data
    except json.JSONDecodeError as e:
        print(f"\nJSON 解析失败: {e}")
        print(f"原始内容片段: {response_text[-200:]}")
        return []


def extract():
    if not os.path.exists(JSON_FOLDER):
        print(f"错误：文件夹 '{JSON_FOLDER}' 不存在。")
        return

    json_files = [f for f in os.listdir(JSON_FOLDER) if f.lower().startswith('table_') and f.lower().endswith('.json')]
    if not json_files:
        print(f"在 '{JSON_FOLDER}' 中未找到任何 table_*.json 文件。")
        return

    print(f"共发现 {len(json_files)} 个 table_*.json 文件，开始处理...")

    for filename in json_files:
        file_path = os.path.join(JSON_FOLDER, filename)
        output_path = file_path.replace("table_", "ZHIPU_")

        if os.path.exists(output_path):
            print(f"跳过文件 {filename}（因为 ZHIPU_*.json 已存在）")
            continue

        response_text = call_zhipu_stream(file_path)
        if not response_text:
            print("API返回为空，跳过。")
            continue

        tables = parse_tables(response_text)
        if tables:
            print(f"提取到 {len(tables)} 个表格")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tables, f, ensure_ascii=False, indent=2)
            print(f"文件 {filename} 处理完成:>")
        else:
            print(f"未提取到表格。")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    extract()

