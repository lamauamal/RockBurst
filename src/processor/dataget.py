"""pdf -> MinerU API -> MinerU_json -> table_json -> ZHIPU API -> ZHIPU_json -> csv -> merge.csv"""
from pdf_json_useMinerU import MUjson
from get_tablejson import get_table_from_json
from json_ZHIPU import extract
from dataclean import clean

def main():
    MUjson() # pdf -> MinerU API
    get_table_from_json() # MinerU_json -> table_json
    extract() # table_json -> ZHIPU_json
    cols = ['σθ', 'σc', 'σt', 'SCF', 'B1', 'B2', 'Wet', 'MR']
    clean(cols) # ZHIPU_json -> csv -> merge.csv

if __name__ == '__main__':
    main()
