"""
全连接神经网络（ANN）分类算法实现模块
基于sklearn多层感知机（MLP）封装，对应课内标准全连接前馈神经网络
遵循BaseAlgorithm统一接口，支持自定义网络结构、激活函数、优化器等超参数
"""
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from .base import BaseAlgorithm

class ANN(BaseAlgorithm):
    """
    标准全连接前馈神经网络（ANN）分类实现类
    继承BaseAlgorithm抽象基类，封装多层感知机核心训练与推理能力
    支持配置隐藏层结构、激活函数、优化方式、正则化等超参数，覆盖课内神经网络核心知识点
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化全连接神经网络模型
        Args:
            config: 神经网络超参数配置字典，支持项参考sklearn.neural_network.MLPClassifier
                    若为None则使用默认配置（两层全连接+ReLU激活+Adam优化）
        """
        # 默认超参数配置：经典两层全连接网络，适配多数分类任务与课内教学基准
        default_config = {
            "hidden_layer_sizes": (64, 32),  # 隐藏层结构：第1层64神经元，第2层32神经元
            "activation": "relu",            # 隐藏层激活函数：relu/tanh/logistic/identity
            "solver": "adam",                # 权重优化器：adam(自适应动量)/sgd(随机梯度下降)/lbfgs
            "alpha": 1e-4,                   # L2正则化惩罚系数，防止过拟合
            "learning_rate_init": 0.001,     # 初始学习率
            "learning_rate": "constant",     # 学习率更新策略
            "max_iter": 200,                 # 最大训练迭代轮数
            "batch_size": "auto",            # 小批量梯度下降的批次大小
            "random_state": 42,              # 随机种子，保证实验可复现
            "early_stopping": False,         # 是否启用早停防止过拟合
            "validation_fraction": 0.1       # 早停时自动划分的验证集比例
        }
        # 合并用户配置，用户配置优先级高于默认值
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)
        # 调用父类初始化，存储配置
        super().__init__(config=merged_config)
        # 实例化全连接神经网络分类器
        self.model = MLPClassifier(**self.config)

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        模型训练：执行前向传播与反向传播，迭代优化网络权重
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选，仅用于效果校验，不参与权重更新）
            y_val: 验证集标签（可选，仅用于效果校验）
        """
        # 统一转换为numpy数组，兼容pandas输入格式
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train
        # 执行模型训练（前向传播+反向传播迭代优化）
        self.model.fit(X_train_arr, y_train_arr)
        # 标记模型训练完成
        self._is_trained = True
        # 若传入验证集，打印验证集准确率
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy (ANN): {val_acc:.4f}")

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        模型预测：前向传播输出样本预测类别
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

    def get_loss_curves(self) -> Tuple[List[float], Optional[List[float]]]:
        """
        获取训练过程中的损失曲线
        Returns:
            train_loss: 训练集每轮损失列表
            val_loss: 验证集每轮损失列表（sklearn MLP默认不记录验证损失，返回None）
        """
        self._check_is_trained()
        # 修复：loss_curve_ 本身就是 list，无需调用 tolist()
        train_loss = self.model.loss_curve_ if hasattr(self.model, "loss_curve_") else []
        val_loss = None  # sklearn MLP原生不记录验证集损失
        return train_loss, val_loss

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        模型评估：计算分类核心指标，与项目内其他算法口径完全统一
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
        持久化模型到本地文件，同时保存模型权重与超参数配置
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
        从本地文件加载模型，完整恢复网络权重、状态与超参数配置
        Args:
            path: 模型文件路径
        """
        load_data = joblib.load(path)
        self.model = load_data["model"]
        self.config = load_data["config"]
        self._is_trained = True