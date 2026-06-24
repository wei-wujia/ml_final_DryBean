"""
SVM算法实现模块
基于BaseAlgorithm抽象基类，默认采用RBF（高斯核），支持核函数灵活配置
"""
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# 导入算法基类
from .base import BaseAlgorithm


class SVMWithRBFKernel(BaseAlgorithm):
    """
    基于RBF（高斯核）的SVM算法实现类
    继承BaseAlgorithm，实现训练、预测、评估、模型持久化等核心接口
    支持通过config配置核函数类型（linear/poly/rbf/sigmoid）及对应超参数
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # 初始化默认配置（优先RBF核）
        default_config = {
            "kernel": "rbf",          # 默认RBF核，可配置为linear/poly/sigmoid
            "C": 1.0,                 # 惩罚系数
            "gamma": "scale",         # RBF/poly/sigmoid核的核系数，scale=1/(n_features*X.var())
            "degree": 3,              # 多项式核的度数（仅poly生效）
            "coef0": 0.0,             # 多项式/sigmoid核的独立项
            "probability": True       # 是否启用概率预测
        }
        # 合并用户配置与默认配置（用户配置覆盖默认）
        merged_config = default_config.copy()
        if config:
            merged_config.update(config)
        
        # 调用父类初始化
        super().__init__(merged_config)
        
        # 初始化SVM模型实例
        self.model: Optional[SVC] = None

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        训练SVM模型
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val/y_val: 验证集（仅用于监控，SVM训练本身不依赖验证集）
        """
        # 初始化SVM模型（基于配置的核函数）
        self.model = SVC(
            kernel=self.config["kernel"],
            C=self.config["C"],
            gamma=self.config["gamma"],
            degree=self.config["degree"],
            coef0=self.config["coef0"],
            probability=self.config["probability"],
            random_state=42  # 固定随机种子保证可复现
        )
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 标记训练完成
        self._is_trained = True
        
        # 若有验证集，打印验证集基础指标（可选）
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy (SVM-{self.config['kernel']}): {val_acc:.4f}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Any:
        """
        模型预测
        Args:
            X: 待预测特征数据
        Returns:
            预测标签（数组）
        """
        # 校验模型是否训练
        self._check_is_trained()
        
        # 转换DataFrame为numpy数组（兼容输入格式）
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_arr)

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        模型评估
        Args:
            X: 评估集特征
            y: 评估集真实标签
        Returns:
            评估指标字典（准确率、精确率、召回率、F1值）
        """
        # 校验模型是否训练
        self._check_is_trained()
        
        # 转换输入格式
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        # 预测
        y_pred = self.model.predict(X_arr)
        
        # 计算多分类兼容的评估指标（macro平均）
        metrics = {
            "accuracy": accuracy_score(y_arr, y_pred),
            "precision": precision_score(y_arr, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_arr, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_arr, y_pred, average="macro", zero_division=0)
        }
        return metrics

    def save_model(self, path: str) -> None:
        """
        持久化模型到本地（使用joblib保存sklearn模型）
        Args:
            path: 模型保存路径（如"models/svm_rbf.pkl"）
        """
        self._check_is_trained()
        # 保存模型实例（包含训练好的参数）
        joblib.dump(self.model, path)
        print(f"SVM-{self.config['kernel']} model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        从本地加载模型
        Args:
            path: 模型文件路径
        """
        # 加载模型
        self.model = joblib.load(path)
        # 标记为已训练
        self._is_trained = True
        print(f"SVM-{self.config['kernel']} model loaded from {path}")