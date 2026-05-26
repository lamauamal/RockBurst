"""基于 auto-sklearn 实现自动化机器学习：（src/model/see_pipelinecomponents.py 显示了所有可用的默认管道， 自定义管道需额外添加）
   1. 数据划分：8：2的训练测试集
   2. 数据处理管道：使用所有可行数据处理方法，在模型外部做数据划分
   3. 特征处理管道：自定义添加特征处理管道为不做特征处理，因为已根据参数物理意义选定了 10 组特征组合方案（可以不预设，让分类器自动选择，只需不设置 "feature_preprocessor"，模型会使用所有可行策略）
   4. 选择了 7 个分类器："decision_tree", "k_nearest_neighbors", "mlp", "adaboost", "libsvm_svc", "random_forest", "extra_trees"
   5. 日志保存在 static/models 下（由于磁盘空间有限，仅保存了测试分数和每种最佳模型，删除了其他日志信息），向模型传入了训练集和测试集（这里即使向 automl 模型传入了测试集，测试集始终是不参与训练的，只是做性能监测）
要实现固定的数据标准化方法为 RobustScaler（相关代码已注释掉）：自定义数据处理器为 NoPreprocessing，表示不做数据处理（数据采集阶段已对样本类别做了从 0 开始的连续整数的统一映射，不需要使用内置的数据处理管道 feature_type）-> 输入automl模型前对数据做 RobustScaler（都按训练集数据来做，避免数据泄露）
"""
# from add_datapreprocessor import NoPreprocessing # 自定义数据处理器
# import autosklearn.pipeline.components.data_preprocessing
# autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing) # 添加自定义数据处理器-不做数据处理（因为后续在 train 函数中做 RobustScaler 标准化）
import shutil

from newleaderboard import AutoSklearnClassifier_leaderboard # 修复源码 leaderboard 因键错误而无法从 runhistory 取出有效 datapreprocessor（实际还取出了 rescaling）

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
# from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy, f1_macro, f1_micro, f1_weighted, log_loss
from sklearn.metrics import roc_auc_score
from smac.optimizer.smbo import SMBO
from typing import Union
from smac.runhistory.runhistory import RunInfo, RunValue
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

load_dotenv()
MERGECSV = os.path.join(project_root, os.getenv("DATA"))

SEED = 42
K_fold = 5
featureslist = {'F-1': ['σθ', 'σc', 'σt', 'Wet'],
                'F-2':['Wet', 'SCF', 'B1'],
                'F-3':['Wet', 'SCF', 'B2'],
                'F-4':['σθ','Wet', 'SCF'],
                'F-5':['σθ','Wet', 'SCF', 'B1'],
                'F-6':['σθ','Wet', 'SCF', 'B2'],
                'F-7':['σθ','Wet', 'SCF', 'σc'],
                'F-8':['σθ', 'σc', 'σt', 'Wet', 'SCF', 'B1', 'B2'],
                'F-9':['σθ', 'σc', 'σt', 'Wet', 'SCF', 'B1'],
                'F-10':['σθ', 'σc', 'σt', 'Wet', 'SCF', 'B2']
                }


def splitdata(datapath, colsname, labelcolname, seed=SEED):  # 按 8：2 划分训练测试集
    data = pd.read_csv(datapath)
    X = data[colsname]
    y = data[labelcolname]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=seed,
        stratify=y,
        shuffle=True
    )

    return X_train, X_test, y_train, y_test


def callback(
        smbo: SMBO,
        run_info: RunInfo,
        result: RunValue,
        time_left: float,
) -> Union[bool, None]:
    if result.cost <= 0.02:
        print("Stopping!")
        print(run_info)
        print(result)
        return False

    return None

def get_pre_score(logpath, automl, y_test):
    """
    从 leaderboard 中定位每种分类器类型的最优模型位置，获取测试集预测结果（建模时需传入测试集），并计算预测指标
    :return: 返回字典的列表 [{
                    'classifier_type': row['type'], # 模型类型
                    'model_id': model_id, # 模型序号，可定位到具体模型位置（.../models/.../runs/seed_modelid_budget）
                    'cv_cost': row['cost'], # 训练 cost,越小越好
                    'cv_rank': row['rank'], # leaderboard 中的排序，按 cost 升序，排名靠前说明模型交叉验证得分高
                    'cv_duration': row['duration'], # 训练时间
                    'data_preprocessors': row['data_preprocessors'], # 数据处理
                    'feature_preprocessors': row['feature_preprocessors'], # 特征处理
                    'balancing_strategy': row['balancing_strategy'], # 平衡测量
                    'config_origin': row['config_origin'] # 超参数优化
                    'balanced_accuracy': b_accuracy,
                    'f1_macro': f1macro,
                    'f1_micro': f1micro,
                    'f1_weighted': f1we,
                    'log_loss': loss,
                    'roc_auc': rocauc
    },{...},...]
    """
    leaderboard = automl.leaderboard1(detailed=True, ensemble_only=False)
    leaderboard = leaderboard.reset_index() # 原来的 model_id 是索引名，经 reset_index 重新建立索引，原来的索引变为新列 model_id
    # print(leaderboard.to_string())
    best_per_type = (leaderboard
                     .sort_values(['type', 'rank'])  # 按 type 和 rank 排序
                     .groupby('type')
                     .first()
                     ) # 按分类器类型分组，获取 cost 最低的模型（leaderboard 默认按 cost 升序）（不能 reset_index，因为后续调用模型需要原有的序列号；索引变为 type）
    # print(best_per_type.to_string())

    results = []
    model_paths = []
    for type, row in best_per_type.iterrows():
        model_id = row['model_id'] # 模型序号，可定位到具体模型位置（.../models/.../runs/seed_modelid_budget）
        seed = automl.automl_._seed
        budget = row['budget']
        # 可以通过源码（backend.py 关于包括模型在内的各种配置的保存和加载）知道日志文件的结构
        model_path = os.path.join(logpath, f".auto-sklearn/runs/{seed}_{model_id}_{budget}/{seed}.{model_id}.{budget}.model")
        pre_path = os.path.join(logpath, f".auto-sklearn/runs/{seed}_{model_id}_{budget}/predictions_test_{seed}_{model_id}_{budget}.npy")
        y_pre_proba = np.load(pre_path, allow_pickle=True) # 获取每个测试样本的类别概率分布
        y_pred = np.argmax(y_pre_proba, axis=1) # 获取预测类别
        # 计算指标 balanced_accuracy, f1_macro, f1_weighted, log_loss, roc_auc
        b_accuracy = balanced_accuracy(y_test, y_pred)
        f1macro = f1_macro(y_test, y_pred)
        f1micro = f1_micro(y_test, y_pred)
        f1we = f1_weighted(y_test, y_pred)
        loss = log_loss(y_test, y_pre_proba)
        rocauc = roc_auc_score(y_test, y_pre_proba, multi_class='ovr', average='weighted')

        re={
            'classifier_type': type,  # 模型类型
            'model_id': model_id,
            'cv_cost': row['cost'],
            'cv_rank': row['rank'],
            'cv_duration': row['duration'],
            'data_preprocessor': row['data_preprocessors'], # 数据处理选项名称（源码中只有一个可用选项 feature_type），数据标准化方法
            'feature_preprocessor': row['feature_preprocessors'],
            'balancing_strategy': row['balancing_strategy'],
            'config_origin': row['config_origin'],
            'balanced_accuracy': b_accuracy,
            'f1_macro': f1macro,
            'f1_micro': f1micro,
            'f1_weighted': f1we,
            'log_loss': loss,
            'roc_auc': rocauc
        }
        results.append(re)
        model_paths.append(model_path)
    pd.DataFrame(results).to_csv(os.path.join(logpath,'results.csv') , index=False) # 保存每种最佳分类器的预测分数到 logpath 下
    for model_path in model_paths:
        shutil.move(model_path, logpath)  # 移动每种最佳分类器到 logpath 下
    # 空间有限，只保存预测分数和每种最佳验证分数的分类器，删除其他日志文件（管道组件即超参数信息都在保存好的模型 .model 中）
    for item_name in os.listdir(logpath):  # 遍历 logpath 下的一级目录名称
        item_path = os.path.join(logpath, item_name)

        if os.path.isfile(item_path):
            _, ext = os.path.splitext(item_name)
            if ext.lower() not in ['.csv', '.model']:
                os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    return results

def ml_models(logsavepath):
    automl = AutoSklearnClassifier_leaderboard(
        time_left_for_this_task=9000, # 搜素模型的最长时间
        per_run_time_limit=1200,  # 训练单个模型的最长时间
        initial_configurations_via_metalearning=0,  # 超参数优化重新开始
        ensemble_class=None,  # 禁用集成构建
        max_models_on_disc=None,  # 保存所有模型
        seed=SEED,
        memory_limit=10240,  # 10GB 内存限制
        include={
            # "data_preprocessor":["NoPreprocessing"] # 不做数据处理，因为 robustscaler 后才传入数据
            "feature_preprocessor": ["no_preprocessing"],  # 不做特征工程，因为已结合参数物理意义选定 10 种特征组合方案
            "classifier": ["decision_tree", "k_nearest_neighbors", "mlp", "adaboost", "libsvm_svc", "random_forest", "extra_trees"]
        },
        resampling_strategy='cv',  # 使用交叉验证
        resampling_strategy_arguments={
            'folds': K_fold,
            'shuffle': True,
            'random_state': SEED
        },
        tmp_folder=logsavepath,  # 存储配置输出和日志文件的文件夹
        delete_tmp_folder_after_terminate=None,  # 不删除 tmp 文件
        n_jobs=-1,  # 使用所有可用的 CPU 核心
        metric=balanced_accuracy,  # 选择平衡准确率为优化指标
        # scoring_functions=[balanced_accuracy,f1_macro,f1_weighted,log_loss,roc_auc], # 将为每个 pipeline 计算的评估指标列表
        get_trials_callback=callback,  # 早停策略
    )

    return automl


def train_pre():
    x_colsname = ['σθ', 'σc', 'σt', 'Wet', 'SCF', 'B1', 'B2']
    y_colname = 'MR'
    X_train, X_test, y_train, y_test = splitdata(MERGECSV, x_colsname, y_colname)

    for key, features in featureslist.items():
        X_train_, X_test_ = X_train[features], X_test[features]
        # automl中不做数据处理时，数据输入前做 robustscaler
        # scaler = RobustScaler()
        # X_train_scaled = scaler.fit_transform(X_train_)
        # X_test_scaled = scaler.transform(X_test_)

        logpath = os.path.join(project_root, f'static/models/{key}_run{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        automl = ml_models(logpath)
        # fit 中测试集用于评估模型随时间变化的性能，本身不参与训练，模型也不会基于测试结果优化指标。总之，这里的测试集只是用作记录的，不参与决策，不会导致数据泄露。
        automl.fit(X_train_, y_train, X_test_, y_test, dataset_name=key)
        # 训练时 k 折交叉验证，每个模型在训练集上拟合 k 次，但不保留训练好的模型，需要用 refit 将 fit 找到的所有模型拟合到完整训练数据
        automl.refit(X_train_, y_train)

        # print(automl.score(X_test_, y_test)) # 只返回最佳单一或集成模型的预测性能
        re = get_pre_score(logpath, automl, y_test)  # 加载每个最佳模型的预测结果
        print(f"{key}:\n{re}")



if __name__ == '__main__':
    train_pre()