"""
XGBoost分类算法实现模块
基于xgboost库封装，遵循BaseAlgorithm统一接口
属于梯度提升树集成学习算法，适配表格型分类任务，泛化能力优异
新增能力：支持训练/验证损失曲线记录，可直接用于可视化
"""
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from .base import BaseAlgorithm

class XGBoost(BaseAlgorithm):
    """
    XGBoost极端梯度提升树分类实现类
    继承BaseAlgorithm抽象基类，实现训练、预测、评估、持久化全量接口
    核心逻辑：串行训练多棵决策树，每棵树拟合上一轮的残差，逐步提升模型效果
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化XGBoost模型
        Args:
            config: XGBoost超参数配置字典，支持项参考xgboost.XGBClassifier
                    若为None则使用工业界通用基准配置
        """
        # 默认超参数配置：平衡效果与训练速度，适配多分类任务
        default_config = {
            "n_estimators": 100,          # 决策树总数量
            "max_depth": 6,               # 单棵树的最大深度，控制模型复杂度
            "learning_rate": 0.1,         # 学习率（步长），越小训练越稳
            "subsample": 0.8,             # 样本随机采样比例，防止过拟合
            "colsample_bytree": 0.8,      # 特征随机采样比例
            "reg_alpha": 0.0,             # L1正则化系数
            "reg_lambda": 1.0,            # L2正则化系数
            "random_state": 42,           # 随机种子，保证实验可复现
            "n_jobs": -1,                 # 并行计算线程数
            "eval_metric": "mlogloss",    # 多分类评估指标（初始化时指定，适配新版xgboost）
            "objective": "multi:softprob",# 显式多分类概率输出
            "num_class": 7                # DryBean数据集固定7类
        }

        # 合并用户配置，用户配置优先级高于默认值
        merged_config = default_config.copy()
        if config is not None:
            merged_config.update(config)

        # 移除废弃参数，消除警告
        merged_config.pop("use_label_encoder", None)

        # 调用父类初始化，存储配置
        super().__init__(config=merged_config)

        # 初始化模型与训练历史容器
        self.model = XGBClassifier(**self.config)
        self.evals_result: Optional[Dict[str, Dict[str, List[float]]]] = None

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        模型训练：串行训练多棵决策树，梯度下降优化残差
        自动记录训练集+验证集的损失曲线，可用于绘制Loss曲线
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选，传入后同步记录验证损失）
            y_val: 验证集标签（可选）
        """
        # 统一转换为numpy数组，兼容pandas输入
        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_arr = y_train.values if isinstance(y_train, pd.Series) else y_train

        # 构建评估集：始终包含训练集，传入验证集则追加
        eval_set = [(X_train_arr, y_train_arr)]
        if X_val is not None and y_val is not None:
            X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
            y_val_arr = y_val.values if isinstance(y_val, pd.Series) else y_val
            eval_set.append((X_val_arr, y_val_arr))

        # ========== 核心修改：适配新版xgboost接口 ==========
        # 移除fit方法中的evals_result参数，训练后通过方法获取
        self.model.fit(
            X_train_arr,
            y_train_arr,
            eval_set=eval_set,
            verbose=False  # 关闭训练过程打印，如需调试可改为True
        )

        # 训练完成后，调用方法获取损失历史
        self.evals_result = self.model.evals_result()
        # ====================================================

        # 标记训练完成
        self._is_trained = True

    def get_loss_curves(self) -> Tuple[List[float], Optional[List[float]]]:
        """
        获取训练过程中的损失曲线（mlogloss）
        Returns:
            train_loss: 训练集每轮损失列表
            val_loss: 验证集每轮损失列表，无验证集时返回None
        """
        self._check_is_trained()
        if not self.evals_result:
            return [], None

        train_loss = self.evals_result["validation_0"]["mlogloss"]
        val_loss = self.evals_result["validation_1"]["mlogloss"] if len(self.evals_result) >= 2 else None
        return train_loss, val_loss

    def get_feature_importance(self) -> np.ndarray:
        """
        获取特征重要性得分（增益权重）
        Returns:
            与特征维度一一对应的重要性数值数组
        """
        self._check_is_trained()
        return self.model.feature_importances_

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
        持久化模型到本地文件，同时保存模型实例、超参数配置与训练历史
        Args:
            path: 模型保存路径，建议后缀为 .pkl
        """
        self._check_is_trained()
        save_data = {
            "model": self.model,
            "config": self.config,
            "evals_result": self.evals_result
        }
        joblib.dump(save_data, path)

    def load_model(self, path: str) -> None:
        """
        从本地文件加载模型，完整恢复模型状态、超参数配置与训练历史
        Args:
            path: 模型文件路径
        """
        load_data = joblib.load(path)
        self.model = load_data["model"]
        self.config = load_data["config"]
        self.evals_result = load_data.get("evals_result", None)
        self._is_trained = True