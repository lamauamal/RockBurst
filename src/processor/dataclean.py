import glob
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

load_dotenv()
CSV_FILE = os.path.join(project_root,os.getenv("CSV_FILE"))
CSV_FOlDER = os.path.dirname(CSV_FILE)


class DataMerge:
    def __init__(self):
        self.alias_map = {
            'σc/σt': 'B1',
            'B1': 'B1',
            'σθ/σc': 'SCF',
            'SCF': 'SCF',
            '(σc-σt)/(σc+σt)': 'B2',
            'B2': 'B2'
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

    def rename_columns(self, df): #将别名统一
        df.columns = df.columns.str.strip()
        rename_dict = {}
        for col in df.columns:
            if col in self.alias_map:
                rename_dict[col] = self.alias_map[col]
        if rename_dict:
            df.rename(columns=rename_dict, inplace=True)
        return df

    def fill_data(self, source_df, target_cols): #计算缺失值
        missing_cols = [col for col in target_cols if col not in source_df.columns] #缺失参数
        for miscol in missing_cols:
            source_df[miscol] = np.nan

            ca_flag = False
            formula_list = self.formulas[miscol] #找到当前缺失参数的计算公式和依赖项
            for func, deps in formula_list:

                if all(dep in source_df.columns for dep in deps): #如果依赖项都存在，则可计算
                    ca_flag = True
                for dep in deps:
                    if dep not in source_df.columns and dep in self.formulas: #如果依赖项缺失，且在公式表中，则计算依赖项
                        missing_cols.append(dep)
                    if dep not in source_df.columns and self.formulas: #如果依赖项缺失且不在公式表中，那么这个缺失参数无法计算
                        break

                if ca_flag: #计算缺失参数
                    try:
                        calu = func(source_df)
                        mask = source_df[miscol].isna()
                        if mask.any():
                            source_df.loc[mask, miscol] = calu[mask]
                            print(f"计算出列：{miscol}")

                    except Exception as e:
                        print(f"计算中出错：{e}")
        df = source_df[needcols]
        return df

    def process_single_file(self, file_path, target_cols):
        try:
            df = pd.read_csv(file_path)
            print(f"正在处理: {os.path.basename(file_path)}")
            recol_df = self.rename_columns(df.copy())
            result_df = self.fill_data(recol_df, target_cols)
            print(f"数据条数：{len(result_df)}\n不为空的列：{result_df.columns[result_df.notna().any()]}")
            return result_df
        except Exception as e:
            print(f"严重错误，跳过文件{file_path} : {str(e)}")
            return None

    def main(self, input_path, target_cols):
        all_data = []
        if os.path.isdir(input_path):
            csv_files = glob.glob(os.path.join(input_path, "*.csv")) #输入是文件夹则获取所有 csv 文件
            if not csv_files:
                print(f"在 '{input_path}' 未找到任何 CSV 文件。")
                return None
            print(f"共发现 {len(csv_files)} 个 CSV 文件，开始合并...")

        elif os.path.isfile(input_path) and input_path.endswith('.csv'):
            csv_files = [input_path]
            print(f"开始处理{csv_files}...")

        else:
            print("错误，输入路径无效。")
            return None

        for file in csv_files:
            df = self.process_single_file(file, target_cols)
            if df is not None and not df.empty:
                all_data.append(df)

        if not all_data:
            print("没有提取到任何有效数据。")
            return None

        final_df = pd.concat(all_data, ignore_index=True)
        final_df.dropna(axis=0, how='any', inplace=True) #去除仍有空值的行
        final_df.drop_duplicates(subset=None, keep='first', inplace=True) #去重
        print(f"成功合并{len(all_data)}个数据，共{len(final_df)}条\n列：{final_df.columns}")
        outputpath = os.path.join(os.path.dirname(CSV_FOlDER), 'merge.csv')
        final_df.to_csv(outputpath, index=False, encoding='utf-8-sig')
        return final_df


if __name__ == '__main__':
    needcols = ['σθ', 'σc', 'σt', 'SCF', 'B1', 'B2', 'Wet', 'MR']
    merge = DataMerge()
    mergedata = merge.main(CSV_FOlDER,needcols)


