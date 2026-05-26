"""查看可用组件"""
from autosklearn.pipeline.components.classification import ClassifierChoice
from autosklearn.pipeline.components.feature_preprocessing import FeaturePreprocessorChoice
from autosklearn.pipeline.components.data_preprocessing import DataPreprocessorChoice
from autosklearn.pipeline.components.data_preprocessing.rescaling import RescalingChoice

print("可用分类器:")
for name in ClassifierChoice.get_components().keys():
    print(f"  • {name}")

print("\n可用特征预处理器:")
for name in FeaturePreprocessorChoice.get_components().keys():
    print(f"  • {name}")

print("\n可用数据预处理器:")
for name in DataPreprocessorChoice.get_components().keys():
    print(f"  • {name}")

print("\n数据预处理器中可用标准化方法:")
for name in RescalingChoice.get_components().keys():
    print(f"  • {name}")

# 可用分类器:
#   • adaboost
#   • bernoulli_nb
#   • decision_tree
#   • extra_trees
#   • gaussian_nb
#   • gradient_boosting
#   • k_nearest_neighbors
#   • lda
#   • liblinear_svc
#   • libsvm_svc
#   • mlp
#   • multinomial_nb
#   • passive_aggressive
#   • qda
#   • random_forest
#   • sgd
#
# 可用特征预处理器:
#   • densifier
#   • extra_trees_preproc_for_classification
#   • extra_trees_preproc_for_regression
#   • fast_ica
#   • feature_agglomeration
#   • kernel_pca
#   • kitchen_sinks
#   • liblinear_svc_preprocessor
#   • no_preprocessing
#   • nystroem_sampler
#   • pca
#   • polynomial
#   • random_trees_embedding
#   • select_percentile_classification
#   • select_percentile_regression
#   • select_rates_classification
#   • select_rates_regression
#   • truncatedSVD
#
# 可用数据预处理器:
#   • feature_type
#
# 数据预处理器中可用标准化方法:
#   • minmax
#   • none
#   • normalize
#   • power_transformer
#   • quantile_transformer
#   • robust_scaler
#   • standardize




