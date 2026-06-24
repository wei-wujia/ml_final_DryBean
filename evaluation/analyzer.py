"""
模型分析模块
自动化实现过拟合分析、鲁棒性分析，输出结构化结论
"""
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .metrics import MetricsCalculator


class ModelAnalyzer:
    """
    模型自动化分析工具类
    提供过拟合程度判定、噪声鲁棒性分析等深度评估能力
    """
    @staticmethod
    def analyze_overfitting(
        model,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        main_metric: str = "accuracy"
    ) -> Dict[str, Union[float, str]]:
        """
        自动化过拟合分析：对比训练集与测试集指标差异，给出拟合程度判定
        Args:
            model: 已训练的模型实例
            X_train: 训练集特征
            y_train: 训练集标签
            X_test: 测试集特征
            y_test: 测试集标签
            main_metric: 核心判定指标
        Returns:
            分析结果字典，包含训练/测试指标、差值、拟合等级、优化建议
        """
        # 计算训练集与测试集指标
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_metrics = MetricsCalculator.calculate_classification_metrics(y_train, train_pred)
        test_metrics = MetricsCalculator.calculate_classification_metrics(y_test, test_pred)
        train_score = train_metrics[main_metric]
        test_score = test_metrics[main_metric]
        gap = train_score - test_score

        # 拟合程度分级
        if gap < 0.02:
            level = "拟合正常"
            suggestion = "模型泛化能力良好，可尝试适当提升模型复杂度以进一步优化效果。"
        elif 0.02 <= gap < 0.05:
            level = "轻微过拟合"
            suggestion = "存在轻度过拟合，可通过增加正则化强度、减少模型复杂度缓解。"
        elif 0.05 <= gap < 0.1:
            level = "中度过拟合"
            suggestion = "过拟合较明显，建议加强正则、引入数据增强或增加训练数据。"
        else:
            level = "严重过拟合"
            suggestion = "过拟合严重，需大幅降低模型复杂度、增加正则惩罚，或扩充训练数据集。"
        
        return {
            f"train_{main_metric}": train_score,
            f"test_{main_metric}": test_score,
            "gap": round(gap, 4),
            "overfitting_level": level,
            "suggestion": suggestion,
            "train_full_metrics": train_metrics,
            "test_full_metrics": test_metrics
        }

    def analyze_noise_robustness(
        self,
        model,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        noise_ratios: List[float] = None,
        metric: str = "accuracy"
    ) -> Dict[str, Union[float, List[float], str]]:
        """
        噪声鲁棒性分析：给测试集特征加入不同强度的高斯噪声，统计性能下降
        Args:
            model: 已训练的模型实例
            X_test: 原始测试集特征
            y_test: 测试集标签
            noise_ratios: 噪声强度列表（相对于特征标准差的比例）
            metric: 核心评估指标
        Returns:
            dict: 包含原始得分、各噪声强度得分列表、鲁棒性等级、结论
        """
        # 默认噪声档位
        if noise_ratios is None:
            noise_ratios = [0.01, 0.05, 0.1, 0.2]
        
        # 统一数据格式，兼容DataFrame/ndarray输入
        X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
        y_arr = y_test.values if isinstance(y_test, pd.Series) else np.array(y_test)
        
        # 计算原始无噪声基准得分
        y_pred = model.predict(X_arr)
        if metric == "accuracy":
            original_score = accuracy_score(y_arr, y_pred)
        else:
            original_score = f1_score(y_arr, y_pred, average="weighted", zero_division=0)
        
        scores = []
        feature_std = np.std(X_arr, axis=0)
        
        # 遍历所有噪声强度，逐次计算得分并保存
        for ratio in noise_ratios:
            noise = np.random.normal(0, feature_std * ratio, size=X_arr.shape)
            X_noisy = X_arr + noise
            
            y_pred_noisy = model.predict(X_noisy)
            if metric == "accuracy":
                score = accuracy_score(y_arr, y_pred_noisy)
            else:
                score = f1_score(y_arr, y_pred_noisy, average="weighted", zero_division=0)
            scores.append(score)
        
        # 以10%噪声强度为基准计算下降幅度
        target_ratio = 0.1
        closest_idx = min(range(len(noise_ratios)), key=lambda i: abs(noise_ratios[i] - target_ratio))
        drop = original_score - scores[closest_idx]
        drop_ratio = drop / original_score if original_score != 0 else 0

        # 鲁棒性等级判定
        if drop_ratio < 0.05:
            level = "鲁棒性优秀"
        elif drop_ratio < 0.15:
            level = "鲁棒性良好"
        elif drop_ratio < 0.3:
            level = "鲁棒性一般"
        else:
            level = "鲁棒性较差"
        
        conclusion = f"在10%特征强度噪声下，{metric}下降{drop_ratio*100:.2f}%，{level}。"
        
        return {
            "original_score": original_score,
            "scores": scores,        # 核心字段：各噪声强度对应的指标值，与noise_ratios一一对应
            "max_drop": original_score - min(scores),
            "drop_ratio": drop_ratio,
            "robustness_level": level,
            "conclusion": conclusion
        }