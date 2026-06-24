"""
可视化绘图模块
实现多算法指标对比、混淆矩阵、训练曲线、鲁棒性分析等评估可视化能力
"""
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

class ModelPlotter:
    """
    模型评估可视化工具类
    封装分类任务常用的评估图表绘制，支持直接保存为图片
    """
    def __init__(self, style: str = "whitegrid", dpi: int = 120, figsize: tuple = (8, 5)) -> None:
        sns.set_style(style)
        self.dpi = dpi
        self.figsize = figsize
        plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans", "WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False

    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        metrics: List[str] = None,
        title: str = "多算法指标对比",
        save_path: Optional[str] = None
    ) -> None:
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1"]
        
        valid_algos = [algo for algo in metrics_dict if all(m in metrics_dict[algo] for m in metrics)]
        if not valid_algos:
            print("⚠️ 无有效指标数据，跳过指标对比图绘制")
            return
        
        df = pd.DataFrame({algo: metrics_dict[algo] for algo in valid_algos}).T[metrics]
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        x = np.arange(len(metrics))
        width = 0.8 / len(df.index)
        
        for i, algo_name in enumerate(df.index):
            offset = (i - len(df.index) / 2 + 0.5) * width
            plt.bar(x + offset, df.loc[algo_name], width=width, label=algo_name)
        
        plt.xticks(x, metrics)
        plt.ylabel("指标数值")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        
        for i, algo_name in enumerate(df.index):
            offset = (i - len(df.index) / 2 + 0.5) * width
            for j, metric in enumerate(metrics):
                value = df.loc[algo_name, metric]
                plt.text(x[j] + offset, value + 0.01, f"{value:.3f}",
                         ha="center", va="bottom", fontsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_confusion_matrix(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        class_names: Optional[List[str]] = None,
        title: str = "混淆矩阵",
        normalize: bool = False,
        save_path: Optional[str] = None
    ) -> None:
        y_true_arr = y_true.values if isinstance(y_true, pd.Series) else np.array(y_true)
        y_pred_arr = y_pred.values if isinstance(y_pred, pd.Series) else np.array(y_pred)
        
        cm = confusion_matrix(y_true_arr, y_pred_arr)
        if normalize:
            cm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
            fmt = ".2f"
        else:
            fmt = "d"
        
        if class_names is None:
            class_names = [f"类别{i}" for i in range(cm.shape[0])]
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"fontsize": 8})
        plt.xlabel("预测标签")
        plt.ylabel("真实标签")
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_learning_curve(
        self,
        train_scores: List[float],
        val_scores: Optional[List[float]] = None,
        metric_name: str = "准确率",
        x_label: str = "迭代轮次",
        title: str = "训练-验证学习曲线",
        save_path: Optional[str] = None
    ) -> None:
        if not train_scores:
            print("⚠️ 无训练分数数据，跳过学习曲线绘制")
            return
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        epochs = range(1, len(train_scores) + 1)
        plt.plot(epochs, train_scores, "o-", label="训练集", linewidth=1.5, markersize=4)
        
        if val_scores is not None and len(val_scores) > 0:
            min_len = min(len(train_scores), len(val_scores))
            plt.plot(epochs[:min_len], val_scores[:min_len], "s-", label="验证集", linewidth=1.5, markersize=4)
        
        plt.xlabel(x_label)
        plt.ylabel(metric_name)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    def plot_loss_curve(self, loss_history, val_loss_history=None, title="训练损失曲线", save_path=None):
        if not loss_history:
            print("⚠️ 无损失数据，跳过损失曲线绘制")
            return
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(loss_history, label="训练集损失", linewidth=1.5)
        
        if val_loss_history is not None and len(val_loss_history) > 0:
            min_len = min(len(loss_history), len(val_loss_history))
            plt.plot(val_loss_history[:min_len], label="验证集损失", linewidth=1.5)
        
        plt.xlabel("迭代轮次")
        plt.ylabel("Loss 值")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()

    def plot_multiclass_roc(self, y_true, y_proba, class_names, title="多分类ROC曲线", save_path=None):
        if len(y_true) != len(y_proba):
            print("⚠️ 真实标签与预测概率长度不匹配，跳过ROC曲线绘制")
            return
        
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        n_classes = len(class_names)
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for i in range(n_classes):
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})", linewidth=1.2)
            except ValueError:
                continue
        
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        plt.title(title)
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()

    def plot_feature_importance(self, importance, feature_names, title="特征重要性排序", save_path=None):
        if len(importance) == 0 or len(feature_names) == 0:
            print("⚠️ 无特征重要性数据，跳过绘制")
            return
        
        sorted_idx = np.argsort(importance)[::-1]
        plt.figure(figsize=(self.figsize[0], self.figsize[1]+1), dpi=self.dpi)
        plt.bar(range(len(importance)), importance[sorted_idx], color=sns.color_palette("Blues_r", len(importance)))
        plt.xticks(range(len(importance)), np.array(feature_names)[sorted_idx], rotation=45, ha="right", fontsize=8)
        plt.ylabel("重要性得分")
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()

    # ========== 修复：速度对比图自动匹配字段 ==========
    def plot_speed_comparison(self, metrics_dict, speed_field=None, title="推理速度对比", save_path=None):
        """
        绘制算法推理速度对比图
        :param metrics_dict: 批量评估返回的指标字典（key=算法名，value=指标键值对）
        :param speed_field: 指定速度字段名，None 则自动匹配
        :param title: 图表标题
        :param save_path: 保存路径
        :return: 是否成功生成
        """
        # 1. 自动匹配速度字段（优先级：avg_time_per_sample_ms > throughput_samples_per_sec）
        if speed_field is None:
            # 先获取第一个算法的指标字段，找速度相关字段
            first_algo = next(iter(metrics_dict.keys()))
            available_fields = metrics_dict[first_algo].keys()
            if "avg_time_per_sample_ms" in available_fields:
                speed_field = "avg_time_per_sample_ms"
                y_label = "单样本推理耗时 (毫秒)"
            elif "throughput_samples_per_sec" in available_fields:
                speed_field = "throughput_samples_per_sec"
                y_label = "吞吐量 (样本/秒)"
            else:
                print("⚠️ 未检测到推理速度指标，跳过速度对比图生成")
                print(f"  当前可用指标字段：{list(available_fields)}")
                return False
        else:
            # 显式指定字段时，校验字段存在
            first_algo = next(iter(metrics_dict.keys()))
            if speed_field not in metrics_dict[first_algo]:
                print(f"⚠️ 指定的速度字段 {speed_field} 不存在，跳过速度对比图生成")
                print(f"  当前可用指标字段：{list(metrics_dict[first_algo].keys())}")
                return False
            # 根据字段名设置y轴标签
            if speed_field == "avg_time_per_sample_ms":
                y_label = "单样本推理耗时 (毫秒)"
            elif speed_field == "throughput_samples_per_sec":
                y_label = "吞吐量 (样本/秒)"
            else:
                y_label = speed_field

        # 2. 提取各算法的速度值
        algorithms = []
        speed_values = []
        for algo_name, metrics in metrics_dict.items():
            algorithms.append(algo_name)
            speed_values.append(metrics[speed_field])

        # 3. 绘图
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        sns.barplot(x=algorithms, y=speed_values, palette="viridis")
        plt.title(title, fontsize=12)
        plt.xlabel("算法", fontsize=10)
        plt.ylabel(y_label, fontsize=10)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # 4. 保存
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        return True

    def plot_joint_loss_curve(
        self,
        loss_dict: Dict[str, List[float]],
        title: str = "多算法训练损失曲线对比",
        x_label: str = "迭代轮次",
        y_label: str = "Loss 值",
        save_path: Optional[str] = None
    ) -> None:
        valid_loss = {k: v for k, v in loss_dict.items() if len(v) > 0}
        if not valid_loss:
            print("⚠️ 无有效损失数据，跳过联合损失曲线绘制")
            return
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for algo_name, loss_list in valid_loss.items():
            plt.plot(range(1, len(loss_list)+1), loss_list, 
                     label=algo_name, linewidth=1.5)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    # ========== 修复：噪声鲁棒性图增强校验与提示 ==========
    def plot_noise_robustness(
        self,
        noise_results: Dict[str, List[float]],
        noise_ratios: List[float],
        title: str = "算法噪声鲁棒性对比",
        metric_name: str = "准确率",
        save_path: Optional[str] = None
    ) -> bool:
        valid_results = {}
        for algo_name, scores in noise_results.items():
            if len(scores) == len(noise_ratios) and len(scores) > 0:
                valid_results[algo_name] = scores
        
        if not valid_results:
            print("⚠️ 无有效噪声鲁棒性数据，跳过鲁棒性对比图生成")
            print(f"  预期噪声档位数量：{len(noise_ratios)}")
            for algo, scores in noise_results.items():
                print(f"  - {algo}: 实际数据长度 {len(scores)}")
            return False
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        for algo_name, scores in valid_results.items():
            plt.plot(noise_ratios, scores, "o-", label=algo_name, linewidth=1.5, markersize=4)
        
        plt.xlabel("噪声强度（标准差比例）")
        plt.ylabel(metric_name)
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        return True