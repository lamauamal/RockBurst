"""MinerU API调用文档：https://mineru.net/apiManage/docs
通过API使用MinerU智能解析功能，将pdf文档解析为包含文本、表格、公式的布局信息Json
本地文件批量上传解析、下载回本地、解压、只保留全局Json文件即layout.json并重命名"""

import shutil
import time
import requests
import zipfile
import io
import os
from dotenv import load_dotenv

load_dotenv()
MINERU_API_KEY = os.getenv('MINERU_API_KEY')
MINERU_API_BASE = os.getenv('MINERU_API_BASE')  #智能解析

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
PDF_FOLDER = os.path.join(project_root, os.getenv('PDF_FOLDER'))
JSON_FILE = os.path.join(project_root, os.getenv('JSON_FILE'))
JSON_FOLDER = os.path.dirname(JSON_FILE)


class MinerUAPIClient:
    def __init__(self, PDF_FOLDER):
        self.token = MINERU_API_KEY
        self.base_url = MINERU_API_BASE
        self.pdf_folder =  PDF_FOLDER
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    def prepare_tasks(self):
        if not os.path.exists(self.pdf_folder):
            print(f"错误: 文件夹 '{self.pdf_folder}' 不存在。")
            return [], set()
        if not os.path.exists(JSON_FOLDER):
            os.makedirs(JSON_FOLDER)

        to_process = []
        existing_files = set()
        for f in os.listdir(JSON_FOLDER):
            if f.lower().endswith('.json') and f.startswith('MinerU_'): #MinerU_*.json
                existing_files.add(f)

        all_files = [f for f in os.listdir(self.pdf_folder) if f.lower().endswith('.pdf')]
        if not all_files:
            print(f"在 '{self.pdf_folder}' 中未找到PDF文件。")
            return [], existing_files

        for filename in all_files:
            name_without_ext = os.path.splitext(filename)[0]
            target_json_name = f"MinerU_{name_without_ext}.json"
            if target_json_name in existing_files:
                print(f"跳过 {filename}（因为 MinerU_*.json 已存在）")
            else:
                to_process.append(filename)

        return to_process, existing_files

    def upload_files_and_submit(self, files_to_process, model_version="vlm"):
        if not files_to_process:
            print("没有需要处理的文件。")
            return None, set()

        files_data = []
        file_paths = []

        for filename in files_to_process:
            full_path = os.path.join(self.pdf_folder, filename)
            name = os.path.splitext(filename)[0]


            files_data.append({
                "name": filename,
                "data_id": name  #使用文件名作为data_id
            })
            file_paths.append(full_path)

        payload = {
            "files": files_data,
            "model_version": model_version,
            "enable_table": True,
            "enable_formula": True,
            "is_ocr": True
        }
        print(f"正在申请 {len(files_data)} 个文件的上传链接...")
        try:
            response = requests.post(
                f"{self.base_url}/file-urls/batch",
                headers=self.headers,
                json=payload
            )
            result = response.json()
            if result.get("code") != 0:
                raise Exception(f"申请链接失败: {result.get('msg')}")

            batch_id = result["data"]["batch_id"]
            upload_urls = result["data"]["file_urls"]
            print(f"申请链接成功，Batch ID: {batch_id}")

            #上传
            for local_path, upload_url in zip(file_paths, upload_urls):
                filename = os.path.basename(local_path)
                print(f"正在上传文件 {filename} ...")
                with open(local_path, 'rb') as f:
                    res = requests.put(upload_url, data=f)
                    if res.status_code == 200:
                       print(f"文件 {filename} 上传成功:>")
                    else:
                       print(f"文件 {filename}上传失败:< {res.status_code}")
            return batch_id, set([f['name'] for f in files_data])

        except Exception as e:
            print(f"错误:上传中发生错误{e}")
            return None, set()


    def get_batch_result(self, batch_id):
        url = f"{self.base_url}/extract-results/batch/{batch_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()

    def deal_zip(self, url, filename):
        name = os.path.splitext(filename)[0]
        target_json_name = f"MinerU_{name}.json"
        target_path = os.path.join(JSON_FOLDER, target_json_name)
        try:
            print(f"正在下载文件 {filename} 解析结果...")
            response = requests.get(url)
            if response.status_code == 200:
                existing_files = set(os.listdir(JSON_FOLDER))

                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    z.extractall(JSON_FOLDER)
                print(f"文件 {filename} 解压完成。")

                current_files = set(os.listdir(JSON_FOLDER))
                new_files = current_files - existing_files

                for item in new_files:
                    item_path = os.path.join(JSON_FOLDER, item)

                    if item == "layout.json":
                        os.rename(item_path, target_path)
                    elif os.path.isdir(item_path): #文件夹则删除
                         shutil.rmtree(item_path)
                    else:
                         os.remove(item_path) #非目标json文件删除
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")


def MUjson():
    client = MinerUAPIClient(PDF_FOLDER)
    files_to_process, _ = client.prepare_tasks()
    if not files_to_process:
        print("所有文件已处理。")
        return

    try:
        batch_id, submitted_filenames = client.upload_files_and_submit(files_to_process, model_version="vlm")
        if not batch_id:
            return
        print(f"正在等待解析完成 (Batch ID: {batch_id})...")
        while True: #轮询结果
            result = client.get_batch_result(batch_id)
            if result.get("code") != 0:
                print(f"查询出错: {result.get('msg')}")
                break

            all_done = True
            for file_result in result["data"]["extract_result"]:
                if file_result["file_name"] not in submitted_filenames:
                    continue

                filename = file_result["file_name"]
                state = file_result["state"]

                if state == "done":
                    print(f"{filename} -> 解析完成")
                elif state == "failed":
                    print(f"{filename} -> 解析失败")
                else:
                    all_done = False
                    progress = file_result.get("extract_progress", {})
                    print(
                        f"{filename} -> 处理中... {progress.get('extracted_pages', 0)}/{progress.get('total_pages', '未知')} 页")

            if all_done:
                print("所有文件解析完成，开始下载并处理 JSON...")
                #下载并处理每个文件的结果
                for item in result["data"]["extract_result"]:
                    if item["state"] == "done":
                        zip_url = item['full_zip_url']
                        file_name = item['file_name']
                        client.deal_zip(zip_url, file_name)
                break

            time.sleep(10)

    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    MUjson()
