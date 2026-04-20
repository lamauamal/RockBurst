import csv
import os
import base64
import json
import time
import requests
from dotenv import load_dotenv
import fitz

load_dotenv()
API_KEY = os.getenv('ZHIPU_API_KEY')
MODEL_NAME = os.getenv('ZHIPU_MODEL_NAME')
API_URL = os.getenv('ZHIPU_API_URL')

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

PDF_FOLDER = os.path.join(project_root,os.getenv('PDF_FOLDER'))
JSON_FILE = os.path.join(project_root, os.getenv('JSON_FILE'))
CSV_FILE = os.path.join(project_root,os.getenv("CSV_FILE"))
PROMPT_FILE = os.path.join(project_root,os.getenv('PROMPT_FILE'))
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    BASE_PROMPT = f.read()


def pdfpages_to_base64list(pdf_path, start_page, end_page):
    doc = fitz.open(pdf_path)
    image_contents = []
    actual_end = min(end_page, len(doc))

    for page_num in range(start_page, actual_end):
        page = doc.load_page(page_num)
        zoom = 1.5
        mat = fitz.Matrix(zoom, zoom) #放大图片提高分辨率
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        base64_str = base64.b64encode(img_data).decode('utf-8')
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_str}"
            }
        })
    doc.close()
    return image_contents, actual_end - start_page


def call_zhipu_stream(images, prompt, retry_count=3):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    content_payload = images + [{"type": "text", "text": prompt}]
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content_payload}],
        "stream": True,  #开启流式（输出思考过程避免timeout）
        "temperature": 0.2,
        "top_p": 0.7
    }

    for attempt in range(retry_count):
        try:
            print(f"发送 ({len(images)} 张图片)... ")
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
        return data.get("tables", [])
    except json.JSONDecodeError as e:
        print(f"\nJSON 解析失败: {e}")
        print(f"原始内容片段: {response_text[-200:]}")
        return []


def process_pdf(pdf_path, output_dir):
    pdf_name = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_name)[0]
    json_path = os.path.join(output_dir, f"{base_name}.json")
    if os.path.exists(json_path):
        print(f"跳过{pdf_name}(因为JSON已存在)")
        return
    print(f"开始处理: {pdf_name}")
    with fitz.open(pdf_path) as doc: #获取总页数
        total_pages = len(doc)
    print(f"总页数: {total_pages}")
    BATCH_SIZE = 2  #每次处理页数，若文件表格内容多，值应调小，避免输出超过规定的最大（产生截断，提取不到表格）

    all_tables = []
    table_header = None
    try:
        for i in range(0, total_pages, BATCH_SIZE):
            start_p = i
            end_p = min(i + BATCH_SIZE, total_pages)
            print(f"正在处理: {start_p + 1} - {end_p}")
            images, count = pdfpages_to_base64list(pdf_path, start_p, end_p)
            prompt = (
                f"{BASE_PROMPT}\n"
                f"【上下文】这些图片对应原 PDF 的第 {start_p + 1} 页到第 {end_p} 页。"
                f"请严格提取其中的表格，并将 'page_num' 标记为真实的页码（如 {start_p + 1}）。"
                f"直接输出 JSON，不要多余解释。"
                f"如果表格没有识别出表头，尝试表头{table_header}")
            response_text = call_zhipu_stream(images, prompt)
            tables = parse_tables(response_text)
            if tables:
                print(f"提取到 {len(tables)} 个表格")
                all_tables.extend(tables)
                if table_header is None:
                    first_table = tables[0]
                    new_header = first_table.get("headers")
                    if new_header and isinstance(new_header, list):
                        table_header = new_header
            else:
                print(f"未提取到表格")

        if not all_tables:
            print("未提取到表格")
        print(f"正在保存 {len(all_tables)} 个表格到 JSON...")
        table_json = {"tables": all_tables}
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(table_json, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"处理 {pdf_name} 时发生错误: {e}")
        import traceback
        traceback.print_exc()


def pdf_json_useZhiPuAI(pdffolder, jsonfile):
    if not os.path.exists(pdffolder):
        print(f"错误:文件夹'{pdffolder}' 不存在。")
        return
    output_dir = os.path.dirname(jsonfile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    pdf_files = [f for f in os.listdir(pdffolder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"在 '{pdffolder}' 中未找到任何 PDF 文件。")
        return
    print(f"共发现 {len(pdf_files)} 个 PDF 文件，开始提取表格...")
    for filename in pdf_files:
        file_path = os.path.join(pdffolder, filename)
        try:
            process_pdf(file_path, output_dir)
        except Exception as e:
            print(f"严重错误，跳过文件 {filename}: {str(e)}")


def process_json(json_file, output_folder):
    json_name = os.path.basename(json_file)
    base_name = os.path.splitext(json_name)[0]
    csv_path = os.path.join(output_folder, f"{base_name}.csv")
    if os.path.exists(csv_path):
        print(f"跳过{csv_path}(因为CSV已存在)")
        return
    print(f"开始处理: {json_name}")

    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not data.get('tables'):
            print(f"在{json_file} 中未找到任何table，跳过。")
            return
        all_headers_set = set()
        tables_data = []
        for table in data['tables']:
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            all_headers_set.update(headers)
            tables_data.append({'headers': headers, 'rows': rows})
        all_headers = list(all_headers_set)
        if not all_headers:
            print(f"在{json_file} 中未找到任何header，跳过。")
            return
        else:print(f"总列数：{len(all_headers)}")

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(all_headers)
            for table in tables_data:
                current_headers = table['headers']
                current_rows = table['rows']
                header_to_index = {h: i for i, h in enumerate(current_headers)}
                for row in current_rows:
                    aligned_row = []
                    for master_header in all_headers:
                        if master_header in header_to_index:
                            idx = header_to_index[master_header]
                            if idx < len(row):
                                aligned_row.append(row[idx])
                            else:
                                aligned_row.append('')
                        else:
                            aligned_row.append('')
                    writer.writerow(aligned_row)
        print(f"成功转换json到{csv_path}\n列:{all_headers}")

    except Exception as e:
        print(f"严重错误，跳过文件 {json_file} : {e}")

def json_to_csv(jsonfolder, csvfile):
    if not os.path.exists(jsonfolder):
        print(f"错误:文件夹'{jsonfolder}' 不存在。")
        return
    output_dir = os.path.dirname(csvfile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = [f for f in os.listdir(jsonfolder) if f.lower().endswith('.json')]
    if not json_files:
        print(f"在 '{jsonfolder}' 中未找到任何 JSON 文件。")
    else:
        print(f"共发现 {len(json_files)} 个 JSON 文件，开始转换...")
        for filename in json_files:
            file_path = os.path.join(jsonfolder, filename)
            try:
                process_json(file_path, output_dir)
            except Exception as e:
                print(f"严重错误，跳过文件 {filename}: {str(e)}")

def main():
    pdf_json_useZhiPuAI(PDF_FOLDER, JSON_FILE)
    JSON_FOlDER = os.path.dirname(JSON_FILE)
    json_to_csv(JSON_FOlDER, CSV_FILE)


if __name__ == "__main__":
    main()