"""将dataJson下的所有result_.json转为.csv
根据输入（期望的）列名（包含力学参数列名和岩爆等级列名）提取合并data下的所有.csv
"""
import csv
import os
import pandas as pd
from dotenv import load_dotenv
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

load_dotenv()
CSV_FILE = os.path.join(project_root,os.getenv("CSV_FILE"))
CSV_FOLDER = os.path.dirname(CSV_FILE)
JSON_FILE = os.path.join(project_root, os.getenv('JSON_FILE'))
JSON_FOLDER = os.path.dirname(JSON_FILE)

class DataMerge:
    def __init__(self):
        self.alias_map = {
            'σc/σt': 'B1', 'B1': 'B1',
            'σθ/σc': 'SCF', 'SCF': 'SCF',
            '(σc-σt)/(σc+σt)': 'B2', 'B2': 'B2',
            'UCS': 'σc', 'Rc': 'σc',
            'Rt': 'σt'
        }
        self.formulas = {
            'SCF': [(lambda df: df['σθ'] / df['σc'], ['σθ', 'σc'])],
            'B1': [(lambda df: df['σc'] / df['σt'], ['σc', 'σt'])],
            'B2': [(lambda df: (df['σc'] - df['σt']) / (df['σc'] + df['σt']), ['σc', 'σt'])],
            'σθ': [(lambda df: df['SCF'] * df['σc'], ['SCF', 'σc'])],
            'σc': [
                (lambda df: df['σθ'] / df['SCF'], ['σθ', 'SCF']),
                (lambda df: df['B1'] * df['σt'], ['B1', 'σt'])
            ],
            'σt': [(lambda df: df['σc'] / df['B1'], ['σc', 'B1'])]
        }

    def rename_columns(self, df):
        df.columns = df.columns.str.strip()
        rename_dict = {}
        for col in df.columns:
            if col in self.alias_map:
                rename_dict[col] = self.alias_map[col]
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
        return df

    def fill_data(self, source_df, target_cols): #计算缺失值
        missing_cols = [col for col in target_cols if col not in source_df.columns]  #缺失参数
        for miscol in missing_cols:
            ca_flag = False
            formula_list = self.formulas[miscol]  #找到当前缺失参数的计算公式和依赖项

            for func, deps in formula_list:
                if all(dep in source_df.columns for dep in deps):  #如果依赖项都存在，则可计算
                    ca_flag = True
                for dep in deps:
                    if dep not in source_df.columns and dep in self.formulas:  # 如果依赖项缺失，但在公式表中，则计算依赖项
                        if dep not in missing_cols:
                            missing_cols.append(dep)
                    if dep not in source_df.columns and self.formulas:  # 如果依赖项不在原数据且不在公式表中，那么这个缺失参数无法计算
                        break

                if ca_flag:  # 计算缺失参数
                    try:
                        calu = func(source_df)
                        source_df[miscol] = calu.round(2)
                        print(f"计算出列：{miscol}")

                    except Exception as e:
                        print(f"计算中出错：{e}")
        df = source_df[target_cols]
        return df

    def process_single_file(self, file_path, target_cols):  # 从单个csv提取数据
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                return None

            print(f"正在处理: {os.path.basename(file_path)}")
            recol_df = self.rename_columns(df.copy())  # 重命名列
            result_df = self.fill_data(recol_df, target_cols)  # 填充数据
            result_df.dropna(how='all')
            if not result_df.empty:
                print(f"数据条数：{len(result_df)}\n不为空的列({len(result_df.columns[result_df.notna().any()])})：{result_df.columns[result_df.notna().any()]}")
            return result_df

        except Exception as e:
            print(f"严重错误，跳过文件 {file_path} : {str(e)}")
            return None

def json_to_csv(json_file, csv_folder): #单个result_文件名.json转为csv
    json_name = os.path.basename(json_file)
    base_name = str(os.path.splitext(json_name)[0]).replace('ZHIPU_', '')
    csv_path = os.path.join(csv_folder, f"{base_name}.csv")
    os.makedirs(csv_folder, exist_ok=True)
    if os.path.exists(csv_path):
        print(f"跳过{csv_path}（因为 CSV 已存在）")
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
        priority_cols = ['No.', 'σθ', 'σc', 'σt', 'SCF', 'B1', 'B2', 'Wet', 'MR', 'H', 'Kv']
        all_headers = [col for col in priority_cols if col in all_headers_set]
        all_headers += [col for col in all_headers_set if col not in all_headers]
        if not all_headers:
            print(f"在{json_file} 中未找到任何header，跳过。")
            return

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

def jsonfloder_to_csv(jsonfolder, csvfile): #result_文件名.json批量转为csv
    if not os.path.exists(jsonfolder):
        print(f"错误: 文件夹'{jsonfolder}' 不存在。")
        return
    output_dir = os.path.dirname(csvfile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    json_files = []
    for filename in os.listdir(JSON_FOLDER):
        full_path = os.path.join(JSON_FOLDER, filename)
        if (os.path.isfile(full_path) and
                filename.startswith('ZHIPU_') and
                filename.endswith('.json')):
            json_files.append(filename)
    if not json_files:
        print(f"在 '{jsonfolder}' 中未找到任何 ZHIPU_*.json 文件。")
        return
    else:
        print(f"共发现 {len(json_files)} 个 ZHIPU_*.json 文件，开始转换...")
        for filename in json_files:
            file_path = os.path.join(jsonfolder, filename)
            try:
                json_to_csv(file_path, output_dir)
            except Exception as e:
                print(f"严重错误，跳过文件 {filename}: {str(e)}")

def csvmerge(needcols, csvfolder_or_csvfile):
    edata = DataMerge()

    if os.path.isfile(csvfolder_or_csvfile) and csvfolder_or_csvfile.endswith('.csv'):
        csv_files = [csvfolder_or_csvfile]
    else:
        if not os.path.exists(csvfolder_or_csvfile):
            print(f"错误:文件夹 '{csvfolder_or_csvfile}' 不存在。")
            return
        csv_files = [f for f in os.listdir(CSV_FOLDER) if f.lower().endswith('.csv')]

    if not csv_files:
        print(f"未找到任何需要处理的 CSV 文件。")
        return
    print(f"共发现 {len(csv_files)} 个 CSV 文件，开始处理...")

    all_data = []
    for filename in csv_files:
        filepath = os.path.join(CSV_FOLDER, filename)
        df = edata.process_single_file(filepath, needcols)
        if df is not None and not df.empty:
            all_data.append(df)

    if not all_data:
        print("没有提取到任何有效数据。")
        return
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.drop_duplicates(subset=final_df.columns, keep='first', inplace=True)
    final_df.dropna(how='any', inplace=True)
    print(f"成功合并 {len(all_data)} 个数据源，最终共 {len(final_df)} 条数据")
    final_df.to_csv(os.path.join(os.path.dirname(CSV_FOLDER),"merge.csv"), index=False, encoding='utf-8-sig')
    return

def clean(needcols):
    jsonfloder_to_csv(JSON_FOLDER, CSV_FILE)
    csvmerge(needcols, CSV_FOLDER)

if __name__ == '__main__':
    cols =['σθ', 'σc', 'σt', 'SCF', 'B1', 'B2', 'Wet', 'MR']
    clean(cols)





