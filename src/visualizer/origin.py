"""通过外部 python 与 origin 交互，实现数据分析和可视化，官方文档：https://docs.originlab.com/originpro/"""
import originpro as op
import os
import sys
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

load_dotenv()
MERGECSV = os.path.join(project_root,os.getenv("DATA"))

if op and op.oext:
    def origin_shutdown_exception_hook(exctype, value, traceback):
        op.exit()
        sys.__excepthook__(exctype, value, traceback)
    sys.excepthook = origin_shutdown_exception_hook
    op.set_show(True)  # 显示 origin 窗口













if __name__ == '__main__':




