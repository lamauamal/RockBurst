# from add_datapreprocessor import NoPreprocessing # 自定义数据处理器
# import autosklearn.pipeline.components.data_preprocessing
# autosklearn.pipeline.components.data_preprocessing.add_preprocessor(NoPreprocessing) # 添加自定义数据处理器-不做数据处理（因为后续在 train 函数中做 RobustScaler 标准化）
import pickle

from add_classifier import XGBoostClassifier,CatBoostClassifier # 自定义分类器
import autosklearn.pipeline.components.classification
autosklearn.pipeline.components.classification.add_classifier(XGBoostClassifier)
autosklearn.pipeline.components.classification.add_classifier(CatBoostClassifier)

import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import RobustScaler
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import balanced_accuracy, f1_macro, f1_weighted, log_loss, roc_auc
from smac.optimizer.smbo import SMBO  # 贝叶斯优化超参数搜索
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
                # 'F-2':['Wet', 'SCF', 'B1'],
                # 'F-3':['Wet', 'SCF', 'B2'],
                # 'F-4':['σθ','Wet', 'SCF'],
                # 'F-5':['σθ','Wet', 'SCF', 'B1'],
                # 'F-6':['σθ','Wet', 'SCF', 'B2'],
                # 'F-7':['σθ','Wet', 'SCF', 'σc'],
                # 'F-8':['σθ', 'σc', 'σt', 'Wet', 'SCF', 'B1', 'B2'],
                # 'F-9':['σθ', 'σc', 'σt', 'Wet', 'SCF', 'B1'],
                # 'F-10':['σθ', 'σc', 'σt', 'Wet', 'SCF', 'B2']
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


def pre_best_classfiers(logpath, automl, X_test, y_test):
    """
    从 leaderboard 中选择每种分类器类型的最优模型，并预测
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
    },{...},...]
    """
    leaderboard = automl.leaderboard(detailed=True, ensemble_only=False)
    leaderboard = leaderboard.reset_index() # 原来的 model_id 是索引名，经 reset_index 重新建立索引，原来的索引变为新列 model_id
    best_per_type = (leaderboard
                     .sort_values(['type', 'rank'])  # 按 type 和 rank 排序
                     .groupby('type')
                     .first()
                     ) # 按分类器类型分组，获取 cost 最低的模型（leaderboard 默认按 cost 升序）（不能 reset_index，因为后续调用模型需要原有的序列号）

    test_metrics = {  # 定义测试指标
        'balanced_accuracy': balanced_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'log_loss': log_loss,
        'roc_auc': roc_auc
    }

    results = []
    for _, row in best_per_type.iterrows():
        model = None

        model_id = row['model_id'] # 模型序号，可定位到具体模型位置（.../models/.../runs/seed_modelid_budget）
        seed = automl.automl_._seed
        budget = row['budget']
        # 可以通过源码（backend.py 关于包括模型在内的各种配置的保存和加载）知道日志文件的结构
        model_path = os.path.join(logpath, f".auto-sklearn/runs/{seed}_{model_id}_{budget}/{seed}.{model_id}.{budget}.model")

        with open(model_path, "rb") as fh:
            model = pickle.load(fh) # 源码 backend.py 中有模型的加载方式，以及各种

        if model is not None:
            for step_name, step in model.steps:
                print(f"\n{'=' * 60}")
                print(f"【{step_name}】")
                print(f"{'=' * 60}")
                print(f"组件类型: {type(step).__name__}")
                print(f"完整类名: {type(step)}")

                # 如果是 Choice 组件，查看其内部选择
                if hasattr(step, 'choice'):
                    choice = step.choice
                    print(f"\n→ 实际选择的组件: {type(choice).__name__}")
                    print(f"  模块: {type(choice).__module__}")

                    # 查看 choice 的参数
                    if hasattr(choice, 'get_params'):
                        params = choice.get_params()
                        print(f"\n  参数列表 ({len(params)} 个):")
                        for param_name, param_value in sorted(params.items()):
                            print(f"    {param_name}: {param_value}")

                    # 特殊检查：DataPreprocessor 的 column_transformer
                    if hasattr(choice, 'column_transformer'):
                        print(f"\n  column_transformer: {choice.column_transformer}")
                        if choice.column_transformer is None:
                            print(f"    ⚠️  警告: column_transformer 为 None（未拟合）")
                        else:
                            print(f"    ✅ 已拟合")
                            # 查看 column_transformer 的详细信息
                            if hasattr(choice.column_transformer, 'transformers_'):
                                print(f"    变换器数量: {len(choice.column_transformer.transformers_)}")

                    # 特殊检查：分类器的参数
                    if 'classifier' in step_name.lower():
                        print(f"\n  【分类器详细信息】")
                        if hasattr(choice, 'estimator'):
                            print(f"    基础估计器: {type(choice.estimator).__name__}")
                            if hasattr(choice.estimator, 'get_params'):
                                est_params = choice.estimator.get_params()
                                print(f"    估计器参数 ({len(est_params)} 个):")
                                for param_name, param_value in list(est_params.items())[:10]:
                                    print(f"      {param_name}: {param_value}")

                # 如果不是 Choice 组件，直接查看参数
                elif hasattr(step, 'get_params'):
                    params = step.get_params()
                    print(f"\n参数列表 ({len(params)} 个):")
                    for param_name, param_value in sorted(params.items()):
                        print(f"  {param_name}: {param_value}")

            # 单独查看最终估计器
            print(f"\n{'=' * 80}")
            print("【最终估计器】")
            print(f"{'=' * 80}")
            if hasattr(model, '_final_estimator'):
                final_est = model._final_estimator
                print(f"类型: {type(final_est).__name__}")
                print(f"完整类名: {type(final_est)}")

                if hasattr(final_est, 'get_params'):
                    params = final_est.get_params()
                    print(f"\n参数 ({len(params)} 个):")
                    for param_name, param_value in sorted(params.items()):
                        print(f"  {param_name}: {param_value}")

            # 检查模型是否真的可以预测
            print(f"\n{'=' * 80}")
            print("【模型状态验证】")
            print(f"{'=' * 80}")

            # 检查关键属性
            check_attributes = [
                'classes_',
                'feature_names_in_',
                'n_features_in_',
                '_final_estimator'
            ]

            for attr in check_attributes:
                has_attr = hasattr(model, attr)
                print(f"{attr}: {'✅ 存在' if has_attr else '❌ 不存在'}")
                if has_attr:
                    value = getattr(model, attr)
                    print(f"  值: {value}")
    #         y_pred = model.predict(X_test)
    #         y_pred_proba = model.predict_proba(X_test)
    #
    #         result_row = {
    #             'classifier_type': row['type'],  # 模型类型
    #             'model_id': model_id,
    #             'cv_cost': row['cost'],
    #             'cv_rank': row['rank'],
    #             'cv_duration': row['duration'],
    #             'data_preprocessors': row['data_preprocessors'],
    #             'feature_preprocessors': row['feature_preprocessors'],
    #             'balancing_strategy': row['balancing_strategy'],
    #             'config_origin': row['config_origin']
    #         }
    #
    #         for metric_name, metric_func in test_metrics.items():  # 计算测试集上的所有指标
    #             if 'log_loss' in metric_name:
    #                 score = metric_func(y_test, y_pred_proba)
    #             else:
    #                 score = metric_func(y_test, y_pred)
    #             result_row[metric_name] = score
    #         results.append(result_row)
    #
    # return results


def ml_models(logsavepath):
    automl = AutoSklearnClassifier(
        time_left_for_this_task=60,  # 搜素模型的最长时间
        per_run_time_limit=10,  # 训练单个模型的最长时间
        initial_configurations_via_metalearning=0,  # 超参数优化重新开始
        ensemble_class=None,  # 禁用集成构建
        max_models_on_disc=None,  # 保存所有模型
        seed=SEED,
        memory_limit=10240,  # 10GB 内存限制
        include={
            # "data_preprocessor":["NoPreprocessing"] # 不做数据处理，因为 robustscaler 后才传入数据
            "feature_preprocessor": ["no_preprocessing"],  # 不做特征工程，因为已选定 10 种特征组合方案
            "classifier": ["decision_tree"]
            # "k_nearest_neighbors", "mlp", "libsvm_svc", "random_forest", "XGBoost", "CatBoost", “extra_trees”
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
        scoring_functions=[balanced_accuracy,f1_macro,f1_weighted,log_loss,roc_auc], # 将为每个 pipeline 计算的评估指标列表
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
        results = pd.DataFrame(pre_best_classfiers(logpath, automl, X_test_, y_test))  # 每个模型的预测性能
        # print(f"{key}:\n{results}")
        # results.to_csv(os.path.join(logpath, 'modelpredictions.csv'), index=False)


if __name__ == '__main__':
    train_pre()
