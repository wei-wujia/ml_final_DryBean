"""
逻辑回归算法实现模块
基于sklearn封装，遵循BaseAlgorithm统一接口，支持二分类/多分类逻辑回归
"""
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from .base import BaseAlgorithm


class LogisticRegression(BaseAlgorithm):
    """
    逻辑回归算法实现类
    继承BaseAlgorithm抽象基类，实现所有核心接口，封装sklearn逻辑回归能力
    支持配置自定义超参数、训练、预测、评估、模型持久化等能力
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化逻辑回归模型
        Args:
            config: 逻辑回归超参数配置字典，支持的配置项参考sklearn.linear_model.LogisticRegression
                    若为None则使用默认配置
        """
        # 定义逻辑回归默认超参数（优先级：用户配置 > 默认配置）
        default_config = {
            "penalty": "l2",          # 正则化类型
            "C": 1.0,                 # 正则化强度（越小正则化越强）
            "max_iter": 100,          # 最大迭代次数
            "random_state": 42,       # 随机种子
            "multi_class": "auto",    # 多分类策略
            "solver": "lbfgs",        # 优化器
            "n_jobs": -1              # 并行计算线程数（-1表示使用所有CPU）
        }

        # 合并用户配置与默认配置
        if config is not None:
            default_config.update(config)

        # 调用父类初始化方法
        super().__init__(config=default_config)

        # 初始化sklearn逻辑回归实例
        self.model = SklearnLogisticRegression(**self.config)

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        模型训练接口（实现基类抽象方法）
        Args:
            X_train: 训练集特征（np.ndarray/pd.DataFrame）
            y_train: 训练集标签（np.ndarray/pd.Series）
            X_val: 验证集特征（可选，暂不参与训练调优，仅预留接口）
            y_val: 验证集标签（可选，暂不参与训练调优，仅预留接口）
        """
        # 数据类型转换：DataFrame -> ndarray（兼容两种输入格式）
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train

        # 执行模型训练
        self.model.fit(X_train_arr, y_train_arr)

        # 标记模型已训练完成
        self._is_trained = True

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        模型预测接口（实现基类抽象方法）：返回类别预测结果
        Args:
            X: 待预测特征数据（np.ndarray/pd.DataFrame）
        Returns:
            np.ndarray: 类别预测结果（形状为[n_samples,]）
        """
        # 校验模型是否已训练
        self._check_is_trained()

        # 数据类型转换
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        # 执行预测
        return self.model.predict(X_arr)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        扩展接口：返回类别概率预测结果（逻辑回归特有）
        Args:
            X: 待预测特征数据（np.ndarray/pd.DataFrame）
        Returns:
            np.ndarray: 类别概率数组（形状为[n_samples, n_classes]）
        """
        self._check_is_trained()
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict_proba(X_arr)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        模型评估接口（实现基类抽象方法）：计算分类核心指标
        Args:
            X: 评估集特征（np.ndarray/pd.DataFrame）
            y: 评估集真实标签（np.ndarray/pd.Series）
        Returns:
            Dict[str, float]: 评估指标字典，包含：
                - accuracy: 准确率
                - precision: 精确率（加权平均）
                - recall: 召回率（加权平均）
                - f1: F1分数（加权平均）
                - roc_auc: ROC-AUC分数（仅二分类有效，多分类返回np.nan）
        """
        self._check_is_trained()

        # 数据类型转换
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        # 获取预测结果
        y_pred = self.model.predict(X_arr)

        # 计算核心评估指标（加权平均适配多分类）
        metrics = {
            "accuracy": round(accuracy_score(y_arr, y_pred), 4),
            "precision": round(precision_score(y_arr, y_pred, average="weighted"), 4),
            "recall": round(recall_score(y_arr, y_pred, average="weighted"), 4),
            "f1": round(f1_score(y_arr, y_pred, average="weighted"), 4)
        }

        # 二分类场景下计算ROC-AUC（多分类跳过并返回nan）
        try:
            y_proba = self.model.predict_proba(X_arr)[:, 1]
            metrics["roc_auc"] = round(roc_auc_score(y_arr, y_proba), 4)
        except (ValueError, IndexError):
            metrics["roc_auc"] = np.nan

        return metrics

    def save_model(self, path: str) -> None:
        """
        模型持久化接口（实现基类抽象方法）：保存模型到本地文件
        Args:
            path: 模型保存路径（如: "./models/logistic_regression.pkl"）
        """
        self._check_is_trained()

        # 保存模型实例+配置（保证加载后状态完整）
        save_dict = {
            "model": self.model,
            "config": self.config
        }
        joblib.dump(save_dict, path)

    def load_model(self, path: str) -> None:
        """
        模型加载接口（实现基类抽象方法）：从本地文件加载模型
        Args:
            path: 模型文件路径
        """
        # 加载模型数据
        load_dict = joblib.load(path)

        # 恢复模型实例和配置
        self.model = load_dict["model"]
        self.config = load_dict["config"]

        # 标记模型已训练
        self._is_trained = True