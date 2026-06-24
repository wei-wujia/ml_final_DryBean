"""
K近邻（KNN）分类算法实现模块
基于sklearn封装，遵循BaseAlgorithm统一接口，支持自定义邻居数、距离度量、权重方式等超参数
"""
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from .base import BaseAlgorithm


class KNN(BaseAlgorithm):
    """
    K近邻分类算法实现类
    继承BaseAlgorithm抽象基类，实现训练、预测、评估、持久化全量接口
    核心逻辑：基于样本间距离度量，选取最近的K个邻居投票决定分类结果
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化KNN模型

        Args:
            config: KNN超参数配置字典，支持项参考sklearn.neighbors.KNeighborsClassifier
                    若为None则使用默认配置
        """
        # KNN默认超参数（通用基准配置）
        default_config = {
            "n_neighbors": 5,          # 近邻样本数量
            "weights": "uniform",      # 投票权重：uniform(等权投票)/distance(距离加权)
            "metric": "minkowski",     # 距离度量方式
            "p": 2,                    # 闵可夫斯基距离参数：p=1曼哈顿距离，p=2欧氏距离
            "n_jobs": -1               # 并行计算线程数，-1使用全部CPU核心
        }

        # 合并用户配置，用户配置优先级高于默认
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)

        # 调用父类初始化，存储配置
        super().__init__(config=merged_config)

        # 实例化KNN分类器
        self.model = KNeighborsClassifier(**self.config)

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        模型训练（KNN为惰性学习算法，训练阶段仅存储训练数据）

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选，仅用于效果校验，不参与训练）
            y_val: 验证集标签（可选，仅用于效果校验，不参与训练）
        """
        # 统一转换为numpy数组，兼容pandas输入
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train

        # KNN训练：存储训练集数据与标签
        self.model.fit(X_train_arr, y_train_arr)

        # 标记训练完成
        self._is_trained = True

        # 验证集可选效果打印
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy (KNN): {val_acc:.4f}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        模型预测：输出样本预测类别

        Args:
            X: 待预测特征数据

        Returns:
            np.ndarray: 预测类别数组，形状为 [n_samples,]
        """
        self._check_is_trained()
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        扩展接口：输出样本属于各类别的概率

        Args:
            X: 待预测特征数据

        Returns:
            np.ndarray: 类别概率数组，形状为 [n_samples, n_classes]
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
        模型评估：计算分类核心指标，与项目内其他算法口径统一

        Args:
            X: 评估集特征
            y: 评估集真实标签

        Returns:
            Dict[str, float]: 评估指标字典，包含准确率、精确率、召回率、F1分数
        """
        self._check_is_trained()

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y

        y_pred = self.model.predict(X_arr)

        # 加权平均适配多分类与样本不平衡场景，zero_division避免零分母报错
        metrics = {
            "accuracy": round(accuracy_score(y_arr, y_pred), 4),
            "precision": round(precision_score(y_arr, y_pred, average="weighted", zero_division=0), 4),
            "recall": round(recall_score(y_arr, y_pred, average="weighted", zero_division=0), 4),
            "f1": round(f1_score(y_arr, y_pred, average="weighted", zero_division=0), 4)
        }
        return metrics

    def save_model(self, path: str) -> None:
        """
        持久化模型到本地文件，同时保存模型实例与配置

        Args:
            path: 模型保存路径，建议后缀为 .pkl
        """
        self._check_is_trained()
        save_data = {
            "model": self.model,
            "config": self.config
        }
        joblib.dump(save_data, path)

    def load_model(self, path: str) -> None:
        """
        从本地文件加载模型，恢复模型状态与配置

        Args:
            path: 模型文件路径
        """
        load_data = joblib.load(path)
        self.model = load_data["model"]
        self.config = load_data["config"]
        self._is_trained = True