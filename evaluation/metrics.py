"""
模型评估指标计算模块
封装分类任务核心指标计算、批量模型评估、推理速度测试等能力
"""
from typing import Dict, List, Union
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class MetricsCalculator:
    """
    分类任务指标计算器
    提供单模型指标计算、批量模型评估、推理速度测速等能力
    """
    @staticmethod
    def calculate_classification_metrics(
        y_true: Union[np.ndarray, pd.Series, list],
        y_pred: Union[np.ndarray, pd.Series, list],
        average: str = "weighted",
        decimal: int = 4
    ) -> Dict[str, float]:
        """
        计算分类任务核心指标：准确率、精确率、召回率、F1分数
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            average: 多分类平均方式，默认weighted适配样本不平衡
            decimal: 小数保留位数
        Returns:
            指标字典
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        metrics = {
            "accuracy": round(accuracy_score(y_true_arr, y_pred_arr), decimal),
            "precision": round(precision_score(y_true_arr, y_pred_arr, average=average, zero_division=0), decimal),
            "recall": round(recall_score(y_true_arr, y_pred_arr, average=average, zero_division=0), decimal),
            "f1": round(f1_score(y_true_arr, y_pred_arr, average=average, zero_division=0), decimal)
        }
        return metrics

    @staticmethod
    def measure_inference_speed(
        model,
        X_test: Union[np.ndarray, pd.DataFrame],
        n_runs: int = 5,
        warmup: int = 1
    ) -> Dict[str, float]:
        """
        测试模型推理速度：单样本平均耗时 + 吞吐量
        Args:
            model: 已训练模型，需实现predict方法
            X_test: 测试集特征
            n_runs: 重复测试次数，取平均值
            warmup: 预热次数，避免首次加载开销影响结果
        Returns:
            速度指标字典
        """
        X_arr = X_test.values if isinstance(X_test, pd.DataFrame) else np.array(X_test)
        n_samples = X_arr.shape[0]

        # 预热运行，消除首次加载/编译开销
        for _ in range(warmup):
            model.predict(X_arr)

        # 多次计时取平均
        total_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model.predict(X_arr)
            end = time.perf_counter()
            total_times.append(end - start)

        avg_total_time = np.mean(total_times)
        avg_time_per_sample_s = avg_total_time / n_samples  # 单样本耗时，单位秒
        avg_time_per_sample_ms = avg_time_per_sample_s * 1000  # 单样本耗时，单位毫秒
        throughput = n_samples / avg_total_time  # 吞吐量，样本/秒

        return {
            "avg_time_per_sample_ms": round(avg_time_per_sample_ms, 6),
            "throughput_samples_per_sec": round(throughput, 2)
        }

    def batch_evaluate_models(
        self,
        model_dict: Dict[str, object],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        measure_speed: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        批量评估多个模型，返回统一格式的指标结果
        Args:
            model_dict: 模型字典 {算法名: 模型实例}
            X_test: 测试集特征
            y_test: 测试集标签
            measure_speed: 是否测试推理速度
        Returns:
            批量评估结果字典 {算法名: {指标名: 指标值}}
        """
        results = {}

        for algo_name, model in model_dict.items():
            try:
                # 1. 计算分类核心指标
                y_pred = model.predict(X_test)
                metrics = self.calculate_classification_metrics(y_test, y_pred)

                # 2. 测试推理速度（所有算法统一执行）
                if measure_speed:
                    try:
                        speed_metrics = self.measure_inference_speed(model, X_test)
                        metrics.update(speed_metrics)
                    except Exception as e:
                        print(f"⚠️  {algo_name} 速度测试失败: {e}")
                        # 失败时填充占位值，避免字段缺失
                        metrics["avg_time_per_sample_ms"] = 0.0
                        metrics["throughput_samples_per_sec"] = 0.0

                results[algo_name] = metrics

            except Exception as e:
                print(f"❌ {algo_name} 评估失败: {e}")
                # 评估失败时保留空结构，避免后续流程报错
                results[algo_name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "avg_time_per_sample_ms": 0.0,
                    "throughput_samples_per_sec": 0.0
                }

        return results