"""从pdf文档的布局信息json提取所有表格json
pdf_info -> para_blocks(type=table) -> blocks -> type=table_caption -> lines -> spans -> content
                                              -> type=table_body -> lines -> spans -> html(单元格跨行 数值见有空格 删空格修复数据)
"""

import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

JSON_FILE = os.path.join(project_root, os.getenv('JSON_FILE'))
JSON_FOLDER = os.path.dirname(JSON_FILE)


def clean_html_spaces(html_content):
    if not html_content:
        return ""

    cleaned = re.sub(r'(>)(\d+\.\d+)(\s+)(\d+)(<)', r'\1\2\4\5', html_content)
    cleaned = re.sub(r'(>)(\d+)(\s+)(\d+)(<)', r'\1\2\4\5', cleaned)
    return cleaned

def extract_tables_from_mineru(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tables = []
    for page in data.get("pdf_info", []):
        page_idx = page.get("page_idx", "Unknown")  #获取页码
        for block in page.get("para_blocks", []): #在当前页中查找 para_blocks
            if block.get("type") == "table":
                table_item = {
                    "page": page_idx,
                    "captions": [], #标准表格有两个表名（中、英）
                    "html": ""
                }

                for sub_block in block.get("blocks", []):
                    sub_type = sub_block.get("type")
                    if sub_type == "table_caption": #表名
                        lines = sub_block.get("lines", [])
                        for line in lines:
                            for span in line.get("spans", []):
                                if span.get("type") == "text":
                                    content = span.get("content", "").strip()
                                    if content:
                                        table_item["captions"].append(content)

                    elif sub_type == "table_body": #表格数据
                        lines = sub_block.get("lines", [])
                        if lines and "spans" in lines[0]:
                            for span in lines[0].get("spans", []):
                                if span.get("type") == "table" and "html" in span:
                                    raw_html = span["html"]
                                    table_item["html"] = clean_html_spaces(raw_html)
                                    break

                if table_item["html"]:
                    tables.append(table_item)

    return tables

def get_table_from_json():
    if not os.path.exists(JSON_FOLDER):
        print(f"错误: 文件夹 '{JSON_FOLDER}' 不存在。")
        return

    filenames = []
    for filename in os.listdir(JSON_FOLDER):
        full_path = os.path.join(JSON_FOLDER, filename)
        if (os.path.isfile(full_path) and
                filename.startswith('MinerU_') and
                filename.endswith('.json')):
            filenames.append(filename)
    print(f"共发现 {len(filenames)} 个 MinerU_*.json 文件，开始提取表格内容为 table_*.json ...")

    for filename in filenames:
        file_path = os.path.join(JSON_FOLDER, filename)
        outputpath = file_path.replace('MinerU', 'table', 1)
        if os.path.exists(outputpath):
            print(f"跳过文件 {filename} （因为 table_*.json 已存在）")
            continue

        print(f"开始处理: {filename}")
        try:
            tables = extract_tables_from_mineru(file_path)
            for i, tbl in enumerate(tables):
                print(f"--- 提取的表格 {i + 1} ---")
                print(f"来源页面: {tbl['page']}")
                print(f"表名: {tbl['captions']}")
                #print(f"数据预览: {str(tbl['html'])}...")

                with open(outputpath, 'w', encoding='utf-8') as f_out:
                    json.dump(tables, f_out, ensure_ascii=False, indent=4)
                    print(f"文件 {filename} 处理完成:>")
        except Exception as e:
            print(f"严重错误，跳过文件 {filename}: {str(e)}")



if __name__ == "__main__":
    get_table_from_json()