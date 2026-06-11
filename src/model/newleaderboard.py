"""源码 leaderboard 函数存在 bug：
leaderboard 函数从 runhistory 中获取数据处理方式存在问题，函数中要取的键是 data_preprocessing，而 runhistory 中实际为 data_preprocessor，这导致 leaderboard 无法返回有效的数据处理管道名称（始终为空列表[]）
该 .py 要修复这个 bug，同时获取 rescaling ->
发现，要获取的 datapreprocessor 和 rescaling 在 runhistory 中的键都包含字符串 ‘data_preprocessor’ 和‘ __choice__’，
不需要在 leaderboard 中新增 rescaling 列，仅改动 data_preprocessing -> data_preprocessor 即可实现，
若要在 leaderboard 中新增列，需修改 leaderboard 和 _leaderboard_columns 函数 ->
这里仅修复 data_preprocessing -> data_preprocessor，使 leaderboard 的 data_preprocessors 列显示该分类器使用的 data_preprocessor 和 rescaling 方法

已向 auto-sklearn 提出（https://github.com/automl/auto-sklearn/pull/1784，仅修复使 data_preprocessors 正确获取内容，不包括 rescaling）
"""
import numpy as np
from autosklearn.estimators import AutoSklearnEstimator
from typing import (
    Iterable,
    Optional,
    Union,
)
import pandas as pd
from typing_extensions import Literal
from autosklearn.automl import AutoMLClassifier, AutoMLRegressor
from autosklearn.metrics import Scorer
from sklearn.base import ClassifierMixin
from sklearn.utils.multiclass import type_of_target
from autosklearn.data.validation import (
    convert_if_sparse,
)


class AutoSklearnEstimator_leaderboard(AutoSklearnEstimator):
    """扩展 AutoSklearnEstimator，添加 leaderboard1（修复 bug）"""

    def leaderboard1(
            self,
            detailed: bool = False,
            ensemble_only: bool = True,
            top_k: Union[int, Literal["all"]] = "all",
            sort_by: str = "cost",
            sort_order: Literal["auto", "ascending", "descending"] = "auto",
            include: Optional[Union[str, Iterable[str]]] = None,
    ) -> pd.DataFrame:

        num_metrics = (
            1
            if self.metric is None or isinstance(self.metric, Scorer)
            else len(self.metric)
        )
        column_types = AutoSklearnEstimator_leaderboard._leaderboard_columns(num_metrics)
        if num_metrics == 1:
            multi_objective_cost_names = []
        else:
            multi_objective_cost_names = [f"cost_{i}" for i in range(num_metrics)]

        # Validation of top_k
        if (
                not (isinstance(top_k, str) or isinstance(top_k, int))
                or (isinstance(top_k, str) and top_k != "all")
                or (isinstance(top_k, int) and top_k <= 0)
        ):
            raise ValueError(
                f"top_k={top_k} must be a positive integer or pass"
                " `top_k`='all' to view results for all models"
            )

        # Validate columns to include
        if isinstance(include, str):
            include = [include]

        if include == ["model_id"]:
            raise ValueError("Must provide more than just `model_id`")

        if include is not None:
            columns = [*include]

            # 'model_id' should always be present as it is the unique index
            # used for pandas
            if "model_id" not in columns:
                columns.append("model_id")

            invalid_include_items = set(columns) - set(column_types["all"])
            if len(invalid_include_items) != 0:
                raise ValueError(
                    f"Values {invalid_include_items} are not known"
                    f" columns to include, must be contained in "
                    f"{column_types['all']}"
                )
        elif detailed:
            columns = column_types["all"]
        else:
            columns = column_types["simple"]

        # Validation of sorting
        if sort_by == "cost":
            sort_by_cost = True
            if num_metrics == 1:
                sort_by = ["cost", "model_id"]
            else:
                sort_by = multi_objective_cost_names + ["model_id"]
        else:
            sort_by_cost = False
            if isinstance(sort_by, str):
                if sort_by not in column_types["all"]:
                    raise ValueError(
                        f"sort_by='{sort_by}' must be one of included "
                        f"columns {set(column_types['all'])}"
                    )
            elif len(set(sort_by) - set(column_types["all"])) > 0:
                too_much = set(sort_by) - set(column_types["all"])
                raise ValueError(
                    f"sort_by='{too_much}' must be in the included columns "
                    f"{set(column_types['all'])}"
                )

        valid_sort_orders = ["auto", "ascending", "descending"]
        if not (isinstance(sort_order, str) and sort_order in valid_sort_orders):
            raise ValueError(
                f"`sort_order` = {sort_order} must be a str in " f"{valid_sort_orders}"
            )

        # To get all the models that were optmized, we collect what we can from
        # runhistory first.
        def additional_info_has_key(rv, key):
            return rv.additional_info and key in rv.additional_info

        model_runs = {}
        for run_key, run_val in self.automl_.runhistory_.data.items():
            if not additional_info_has_key(run_val, "num_run"):
                continue
            else:
                model_key = run_val.additional_info["num_run"]
                model_run = {
                    "model_id": run_val.additional_info["num_run"],
                    "seed": run_key.seed,
                    "budget": run_key.budget,
                    "duration": run_val.time,
                    "config_id": run_key.config_id,
                    "start_time": run_val.starttime,
                    "end_time": run_val.endtime,
                    "status": str(run_val.status),
                    "train_loss": run_val.additional_info["train_loss"]
                    if additional_info_has_key(run_val, "train_loss")
                    else None,
                    "config_origin": run_val.additional_info["configuration_origin"]
                    if additional_info_has_key(run_val, "configuration_origin")
                    else None,
                }
                if num_metrics == 1:
                    model_run["cost"] = run_val.cost
                else:
                    for cost_idx, cost in enumerate(run_val.cost):
                        model_run[f"cost_{cost_idx}"] = cost
                model_runs[model_key] = model_run

        # Next we get some info about the model itself
        model_class_strings = {
            AutoMLClassifier: "classifier",
            AutoMLRegressor: "regressor",
        }
        model_type = model_class_strings.get(self._get_automl_class(), None)
        if model_type is None:
            raise RuntimeError(f"Unknown `automl_class` {self._get_automl_class()}")

        # A dict mapping model ids to their configurations
        configurations = self.automl_.runhistory_.ids_config

        for model_id, run_info in model_runs.items():
            config_id = run_info["config_id"]
            run_config = configurations[config_id]._values

            run_info.update(
                {
                    "balancing_strategy": run_config.get("balancing:strategy", None),
                    "type": run_config[f"{model_type}:__choice__"],
                    "data_preprocessors": [
                        value
                        for key, value in run_config.items()
                        if "data_preprocessor" in key and "__choice__" in key # 修复 data_preprocessing -> data_preprocessor，会得到 datapreprocessor 和 rescaling
                    ],
                    "feature_preprocessors": [
                        value
                        for key, value in run_config.items()
                        if "feature_preprocessor" in key and "__choice__" in key
                    ],
                }
            )

        # Get the models ensemble weight if it has one
        for (
                    _,
                    model_id,
                    _,
            ), weight in self.automl_.ensemble_.get_identifiers_with_weights():

            # We had issues where the model's in the ensembles are not in the runhistory
            # collected. I have no clue why this is but to prevent failures, we fill
            # the values with NaN
            if model_id not in model_runs:
                model_run = {
                    "model_id": model_id,
                    "seed": pd.NA,
                    "budget": pd.NA,
                    "duration": pd.NA,
                    "config_id": pd.NA,
                    "start_time": pd.NA,
                    "end_time": pd.NA,
                    "status": pd.NA,
                    "train_loss": pd.NA,
                    "config_origin": pd.NA,
                    "type": pd.NA,
                }
                if num_metrics == 1:
                    model_run["cost"] = pd.NA
                else:
                    for cost_idx in range(num_metrics):
                        model_run[f"cost_{cost_idx}"] = pd.NA
                model_runs[model_id] = model_run

            model_runs[model_id]["ensemble_weight"] = weight

        # Filter out non-ensemble members if needed, else fill in a default
        # value of 0 if it's missing
        if ensemble_only:
            model_runs = {
                model_id: info
                for model_id, info in model_runs.items()
                if ("ensemble_weight" in info and info["ensemble_weight"] > 0)
            }
        else:
            for model_id, info in model_runs.items():
                if "ensemble_weight" not in info:
                    info["ensemble_weight"] = 0

        # `rank` relies on `cost` so we include `cost`
        # We drop it later if it's not requested
        if "rank" in columns:
            if num_metrics == 1 and "cost" not in columns:
                columns = [*columns, "cost"]
            elif num_metrics > 1 and any(
                    cost_name not in columns for cost_name in multi_objective_cost_names
            ):
                columns = columns + list(multi_objective_cost_names)

        # Finally, convert into a tabular format by converting the dict into
        # column wise orientation.
        dataframe = pd.DataFrame(
            {
                col: [run_info[col] for run_info in model_runs.values()]
                for col in columns
                if col != "rank"
            }
        )

        # Give it an index, even if not in the `include`
        dataframe.set_index("model_id", inplace=True)

        # Add the `rank` column if needed
        # requested by the user
        if "rank" in columns:
            if num_metrics == 1:
                dataframe.sort_values(by="cost", ascending=True, inplace=True)
            else:
                dataframe.sort_values(by="cost_0", ascending=True, inplace=True)
            dataframe.insert(
                column="rank",
                value=range(1, len(dataframe) + 1),
                loc=list(columns).index("rank") - 1,
            )  # account for `model_id`

        # Decide on the sort order depending on what it gets sorted by
        descending_columns = ["ensemble_weight", "duration"]
        if sort_order == "auto":
            ascending_param = [
                False if sby in descending_columns else True for sby in sort_by
            ]
        else:
            ascending_param = False if sort_order == "descending" else True

        # Sort by the given column name, defaulting to 'model_id' if not present
        if (
                (not sort_by_cost and len(set(sort_by) - set(dataframe.columns)) > 0)
                or (sort_by_cost and "cost" not in dataframe.columns)
                or (
                sort_by_cost
                and any(
            cost_name not in dataframe.columns
            for cost_name in multi_objective_cost_names
        )
        )
        ):
            self.automl_._logger.warning(
                f"sort_by = '{sort_by}' was not present"
                ", defaulting to sort on the index "
                "'model_id'"
            )
            sort_by = "model_id"
            sort_by_cost = False
            ascending_param = True

        # Single objective
        if sort_by_cost:
            dataframe.sort_values(
                by=sort_by, ascending=[True] * len(sort_by), inplace=True
            )
        else:
            dataframe.sort_values(by=sort_by, ascending=ascending_param, inplace=True)

        if num_metrics == 1:
            if "cost" not in columns and "cost" in dataframe.columns:
                dataframe.drop("cost", inplace=True)
        else:
            for cost_name in multi_objective_cost_names:
                if cost_name not in columns and cost_name in dataframe.columns:
                    dataframe.drop(cost_name, inplace=True)

        # Lastly, just grab the top_k
        if top_k == "all" or top_k >= len(dataframe):
            top_k = len(dataframe)

        dataframe = dataframe.head(top_k)

        return dataframe

class AutoSklearnClassifier_leaderboard(AutoSklearnEstimator_leaderboard, ClassifierMixin):
    """This class implements the classification task."""

    def fit(self, X, y, X_test=None, y_test=None, feat_type=None, dataset_name=None):
        """Fit *auto-sklearn* to given training set (X, y).

        Fit both optimizes the machine learning models and builds an ensemble
        out of them.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target classes.

        X_test : array-like or sparse matrix of shape = [n_samples, n_features]
            Test data input samples. Will be used to save test predictions for
            all models. This allows to evaluate the performance of Auto-sklearn
            over time.

        y_test : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Test data target classes. Will be used to calculate the test error
            of all models. This allows to evaluate the performance of
            Auto-sklearn over time.

        feat_type : list, optional (default=None)
            List of str of `len(X.shape[1])` describing the attribute type.
            Possible types are `Categorical` and `Numerical`. `Categorical`
            attributes will be automatically One-Hot encoded. The values
            used for a categorical attribute must be integers, obtained for
            example by `sklearn.preprocessing.LabelEncoder
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html>`_.

        dataset_name : str, optional (default=None)
            Create nicer output. If None, a string will be determined by the
            md5 hash of the dataset.

        Returns
        -------
        self
        """
        # AutoSklearn does not handle sparse y for now
        y = convert_if_sparse(y)

        # Before running anything else, first check that the
        # type of data is compatible with auto-sklearn. Legal target
        # types are: binary, multiclass, multilabel-indicator.
        target_type = type_of_target(y)
        supported_types = ["binary", "multiclass", "multilabel-indicator"]
        if target_type not in supported_types:
            raise ValueError(
                "Classification with data of type {} is "
                "not supported. Supported types are {}. "
                "You can find more information about scikit-learn "
                "data types in: "
                "https://scikit-learn.org/stable/modules/multiclass.html"
                "".format(target_type, supported_types)
            )

        # remember target type for using in predict_proba later.
        self.target_type = target_type

        super().fit(
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
            feat_type=feat_type,
            dataset_name=dataset_name,
        )

        # After fit, a classifier is expected to define classes_
        # A list of class labels known to the classifier, mapping each label
        # to a numerical index used in the model representation our output.
        self.classes_ = self.automl_.InputValidator.target_validator.classes_

        return self

    def predict(self, X, batch_size=None, n_jobs=1):
        """Predict classes for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_labels]
            The predicted classes.
        """
        return super().predict(X, batch_size=batch_size, n_jobs=n_jobs)

    def predict_proba(self, X, batch_size=None, n_jobs=1):
        """Predict probabilities of classes for all samples X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]

        batch_size : int (optional)
            Number of data points to predict for (predicts all points at once
            if ``None``.
        n_jobs : int

        Returns
        -------
        y : array of shape = [n_samples, n_classes] or [n_samples, n_labels]
            The predicted class probabilities.
        """
        pred_proba = super().predict_proba(X, batch_size=batch_size, n_jobs=n_jobs)

        # Check if all probabilities sum up to 1.
        # Assert only if target type is not multilabel-indicator.
        if self.target_type not in ["multilabel-indicator"]:
            assert np.allclose(
                np.sum(pred_proba, axis=1), np.ones_like(pred_proba[:, 0])
            ), "prediction probability does not sum up to 1!"

        # Check that all probability values lie between 0 and 1.
        assert (pred_proba >= 0).all() and (
                pred_proba <= 1
        ).all(), "found prediction probability value outside of [0, 1]!"

        return pred_proba

    def _get_automl_class(self):
        return AutoMLClassifier

