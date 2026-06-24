"""
算法基类模块
定义所有算法的统一调用规范，所有具体算法实现必须继承该基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd


class BaseAlgorithm(ABC):
    """
    算法抽象基类
    约定训练、预测、评估、持久化四大核心能力接口，屏蔽不同算法的实现差异
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        算法初始化

        Args:
            config: 算法超参数配置字典，不同算法可自定义配置项
        """
        # 算法配置（深拷贝避免外部修改影响内部状态）
        self.config: Dict[str, Any] = config.copy() if config is not None else {}
        # 算法模型实例，由子类具体实现赋值
        self.model: Any = None
        # 模型训练状态标记
        self._is_trained: bool = False

    def _check_is_trained(self) -> None:
        """
        内部校验：验证模型是否已完成训练
        预测、评估、保存前调用，避免未训练直接调用下游方法
        """
        if not self._is_trained:
            raise RuntimeError(
                "Model is not trained yet. Please call `fit()` before this operation."
            )

    @abstractmethod
    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> None:
        """
        模型训练（抽象方法，子类必须实现）

        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Any:
        """
        模型预测（抽象方法，子类必须实现）

        Args:
            X: 待预测特征数据

        Returns:
            预测结果，格式由具体算法定义
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """
        模型评估（抽象方法，子类必须实现）

        Args:
            X: 评估集特征
            y: 评估集真实标签

        Returns:
            评估指标字典，key 为指标名，value 为指标数值
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        持久化模型到本地文件（抽象方法，子类必须实现）

        Args:
            path: 模型文件保存路径
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        从本地文件加载模型（抽象方法，子类必须实现）

        Args:
            path: 模型文件路径
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """获取当前算法配置（返回副本，防止外部篡改）"""
        return self.config.copy()