# 目录结构
```text
RockBurst/
├── src/                              # 源代码目录
│   ├── processor/                     # ETL 代码
│   │   ├── pdf_json_useMinerU.py       # 调用 MinerU 文档解析功能，将论文 pdf 解析为 Json 数据
│   │   ├── get_tablejson.py            # 从包含文本、表格、公式及布局信息的论文 Json 中提取表格数据
│   │   ├── json_ZHIPU.py               # 调用 glm-4.6v，结合 Prompt 从表格 Json 提取岩爆数据，并输出结构化 Json
│   │   ├── dataclean.py                # 将每个岩爆数据 json 存储为 csv，并按输入的期望列合并多个数据为最终的原始数据 merge.csv
│   │   └── dataget.py                  # 串联以上每个模块功能，形成完整的自动 ETL 流程 
│   └── model/                         # 模型相关代码        
│       ├── add_datapreprocessor.py     # 自定义符合 auto-sklearn 框架的数据处理管道
│       ├── auto_classifier.py          # 基于 auto-sklearn 框架搭建模型，并进行训练预测
│       ├── nweleaderboard.py           # auto-sklearn 源码 leaderboard 函数存在 bug，继承原函数修正问题
│       ├── see_pipelinecomponents.py   # 显示框架内置管道
│       └── F14_pre.py                  # 使用训练好的模型
├── static/                           # 数据、图表和训练日志目录
│   ├── data/                          # csv 数据目录
│   ├── dataJson/                      # Json 数据目录，包括 pdf 完整解析 Json（MinerU_*）、表格 Json（table_*）、结构化 Json（ZHIPU_*）
│   ├── datapdf/                       # pdf 文档目录
│   ├── models/                        # 训练日志（只保留了每种算法的最佳模型 .model 和这些模型的性能汇总表 results.csv，需要完整日志信息可到 auto_classifier.py 中设置）
│   ├── plt/                           # 数据分析和性能分析图表（数据分析及相关图表使用 origin）
│   ├── merge.csv                      # 模型输入数据
│   └── Prompt.txt                     # 用于从表格 Json 中提取原始数据并结构化输出的提示语
├── .env                              # API KEY、关键目录等配置信息，需新建
├── environment.yml                   # 项目环境信息   
└── README.md                         # 项目简介                        
```

# 配置信息

## 环境配置
- 操作系统：WSL2
- Linux 发行版：Ubuntu 24.04
- 环境管理：conda
- 环境配置文件：environment.yml

## 环境变量配置
在项目根目录下新建 .env，填入你的 API 密钥:
```ini
# 智谱 AI
ZHIPU_API_KEY=YourKey
ZHIPU_MODEL_NAME=glm-4.6v
ZHIPU_API_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions

# MinerU
MINERU_API_BASE=https://mineru.net/api/v4
MINERU_API_KEY=YourKey

# 路径
PDF_FOLDER=./static/datapdf
JSON_FILE=./static/dataJson/example.json
CSV_FILE=./static/data/example.csv
PROMPT_FILE=./static/Prompt.txt
DATA=./static/merge.csv
```

# 项目概述
本项目源于 2023 年下半年北山高放废物地质处置库开挖工程中的岩爆预测需求，当时没有实测工程数据，只做了理论模型，加上工程验证会更加完整。
当前版本对原始流程进行了重构和优化，主要体现在以下方面：
- **1.重构 ETL 流程，实现数据自动化处理**：摒弃传统人工逐篇扫描表格的方式，采用 MinerU 解析多源 PDF 文献，结合 glm-4.6v 大模型进行数据抽取与结构化，显著提升数据构建效率与准确性；
- **2.完善数据分析流程**：系统开展数据分布特征分析，开展箱型图异常值识别与可视化、Kruskal-Wallis 检验、以及 Spearman 秩相关系数分析；
- **3.引入 AutoML 框架，实现端到端自动化建模**：基于 auto-sklearn 构建集成化机器学习管道，将数据预处理、特征工程、模型训练与超参数调优无缝衔接，并利用并行计算加速搜索过程，有效避免人工调参偏差，提升模型性能与开发效率。

针对岩爆预测中存在的小样本（仅数百条样本）和数据类别不平衡问题，项目构建了一套基于机器学习的岩爆等级预测方法。通过 MinerU 与 glm-4.6v 结合的自动化 ETL 流程，从 7 篇学术论文中提取 389 条岩爆样本数据，建立了标准化数据集；结合岩爆机理与数据分析结果，设计了 14 组具有工程解释性的特征组合方案；基于 auto-sklearn 自动机器学习框架对比评估了 7 种机器学习算法，最终确定特征组合 F-14[σθ, σt, Wet, B1] 与 AdaBoost 模型为最优配置，该模型在测试集上取得了最优性能，平衡准确率为 88.5%。
该项目成果可为深部地下工程岩爆风险评估提供可靠的技术支撑，实现岩爆倾向性的早期识别与分级预警，为施工安全决策提供科学依据。

# 展望
项目仍有优化空间：
- **1.数据扩充与智能检索**：充分利用大模型的语义搜索与文献挖掘能力，系统性地检索和整合国内外岩爆相关学术论文与工程案例数据，扩充训练样本规模，缓解小样本问题对模型泛化能力的制约；
- **2.数据分析与可视化增强**：目前数据分析是直接在 origin 中进行，后续增加 python 外部调用 origin 接口；
- **3.特征工程自动化**：充分利用 auto-sklearn 框架内置的特征选择管道，实现特征筛选的自动化与智能化，减少人工干预，提升特征工程效率；
- **4.多模型集成优化**：当前 AdaBoost 模型虽在综合性能上表现最优，但针对不同岩爆等级的预测能力存在差异，KNN 在无岩爆类别、AdaBoost 在轻微岩爆类别、ET 在中等岩爆类别、DT 与 MLP 在强岩爆类别上分别表现最佳。后续可构建基于类别特异性的多模型集成策略，而非单一依赖 AdaBoost 模型，以进一步提升各等级的预测精度；
- **5.模型精细化调优**：以 auto-sklearn 得到的最优模型为基准，开展针对性的超参数精细化调优工作，探索更优的模型配置；
- **6.理论判据对比验证**：引入经典岩爆理论判据作为工程实践的参考基准，与机器学习模型预测结果进行对比分析，在实际工程应用场景中，结合具体地质条件进行针对性对比，为模型预测结果提供理论支撑。注意，由于理论判据往往基于特定地质背景得出，而本项目数据来源的地质环境可能不同，直接在原始数据集上验证理论判据效果有限。

