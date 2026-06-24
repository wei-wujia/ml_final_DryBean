"""
DryBean 机器学习算法对比实验 主入口程序
纯命令行调用，全自动化完成数据加载、模型训练、批量评估、深度分析、可视化输出
仅使用 DryBeanDataset 目录下的 Clean 干净数据集
使用方式：
    1. 运行全部算法 + 全部分析 + 全部图表：
       python main.py
    2. --algorithms 指定算法 + 保存模型：
       python main.py --algorithms logistic_regression,xgboost --save-model
    3. 跳过绘图与深度分析：
       python main.py --no-plot --no-analysis
    4. 查看全部参数说明：
       python main.py -h
"""
import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# ===================== 项目模块导入 =====================
from data_utils.loader import DataLoader
from data_utils import TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH
from data_utils.preprocess import DataPreprocessor

from algorithms import (
    LogisticRegression,
    SVMWithRBFKernel,
    KNN,
    ANN,
    XGBoost
)

from evaluation import MetricsCalculator, ModelPlotter, ModelAnalyzer

# ===================== 全局常量 =====================
ALGORITHM_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "svm": SVMWithRBFKernel,
    "knn": KNN,
    "ann": ANN,
    "xgboost": XGBoost
}

CLASS_NAMES = ["SEKER", "BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SIRA"]

# ===================== 工具函数 =====================
def load_config(config_path: str) -> dict:
    """加载YAML配置文件（增加容错处理）"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path.absolute()}")
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # 补充默认配置（防止键缺失）
        config.setdefault("global", {"random_state": 42})
        config.setdefault("evaluation", {"main_metric": "accuracy", "measure_inference_speed": True, "noise_std_ratios": [0.0, 0.1, 0.2, 0.3]})
        config.setdefault("algorithms", {})
        config.setdefault("paths", {"model_save_dir": "./saved_models"})
        config.setdefault("data", {"standardize": True})
        config.setdefault("visualization", {"style": "whitegrid", "dpi": 120, "figsize_width": 8, "figsize_height": 5})
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"配置文件解析错误: {e}")

def set_global_seed(seed: int) -> None:
    """设置全局随机种子，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def ensure_dir(path: str) -> None:
    """确保目录存在，不存在则创建"""
    Path(path).mkdir(parents=True, exist_ok=True)

def validate_algorithm_list(algo_list: list) -> None:
    """校验算法名称合法性"""
    invalid = [a for a in algo_list if a not in ALGORITHM_REGISTRY]
    if invalid:
        raise ValueError(
            f"不支持的算法: {invalid}\n支持的算法列表: {list(ALGORITHM_REGISTRY.keys())}"
        )

# ===================== 主流程 =====================
def main():
    # ---------- 1. 命令行参数解析 ----------
    parser = argparse.ArgumentParser(
        description="DryBean(Clean) 数据集多算法分类对比实验主程序",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    base_group = parser.add_argument_group("基础配置")
    base_group.add_argument("--config", type=str, default="configs/default.yaml", help="YAML配置文件路径")
    base_group.add_argument("--output-dir", type=str, default="./results", help="实验结果输出根目录")

    algo_group = parser.add_argument_group("算法配置")
    algo_group.add_argument("--algorithms", type=str, default="logistic_regression,svm,knn,ann,xgboost",
                            help="指定运行的算法，逗号分隔")
    algo_group.add_argument("--save-model", action="store_true", help="是否持久化模型")

    eval_group = parser.add_argument_group("评估开关")
    eval_group.add_argument("--no-plot", action="store_true", help="跳过可视化图表生成")
    eval_group.add_argument("--no-analysis", action="store_true", help="跳度过拟合、鲁棒性深度分析")

    args = parser.parse_args()

    # ---------- 2. 初始化配置与环境 ----------
    print("=" * 60)
    print("【1/6】加载配置与初始化环境")
    print("=" * 60)

    config = load_config(args.config)
    set_global_seed(config["global"]["random_state"])
    main_metric = config["evaluation"]["main_metric"]
    noise_ratios = config["evaluation"]["noise_std_ratios"]  # 修复：移到全局作用域，避免未定义错误

    algo_list = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    if not algo_list:
        raise ValueError("未指定任何算法，请通过 --algorithms 参数指定")
    validate_algorithm_list(algo_list)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    plot_dir = os.path.join(run_output_dir, "plots")
    ensure_dir(run_output_dir)
    if not args.no_plot:
        ensure_dir(plot_dir)

    print(f"配置文件: {Path(args.config).absolute()}")
    print(f"数据集: Clean 干净数据集")
    print(f"运行算法: {algo_list}")
    print(f"结果输出目录: {Path(run_output_dir).absolute()}")
    print(f"随机种子: {config['global']['random_state']}")
    print(f"核心评估指标: {main_metric}")
    print(f"噪声档位: {noise_ratios}")

    # ---------- 3. 数据加载与预处理 ----------
    print("\n" + "=" * 60)
    print("【2/6】加载并预处理数据集")
    print("=" * 60)

    try:
        data_loader = DataLoader(
            train_path=TRAIN_CSV_PATH,
            val_path=VAL_CSV_PATH,
            test_path=TEST_CSV_PATH
        )
        train_data, val_data, test_data = data_loader.load_all()
        X_train, y_train = train_data
        X_val, y_val = val_data
        X_test, y_test = test_data
    except Exception as e:
        raise RuntimeError(f"数据集加载失败: {e}")

    feature_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"feature_{i}" for i in range(X_train.shape[1])]

    scaler_type = "standard" if config["data"]["standardize"] else "minmax"
    preprocessor = DataPreprocessor(scaler_type=scaler_type, label_names=CLASS_NAMES)

    X_train = preprocessor.fit_transform_features(X_train)
    X_val = preprocessor.transform_features(X_val)
    X_test = preprocessor.transform_features(X_test)

    y_train = preprocessor.encode_labels(y_train)
    y_val = preprocessor.encode_labels(y_val)
    y_test = preprocessor.encode_labels(y_test)

    print(f"训练集样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}")
    print(f"验证集样本数: {X_val.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    print(f"类别数: {len(np.unique(y_test))}")

    # ---------- 4. 模型初始化与批量训练 ----------
    print("\n" + "=" * 60)
    print("【3/6】初始化并批量训练算法模型")
    print("=" * 60)

    model_dict = {}
    algo_configs = config["algorithms"]

    for algo_name in algo_list:
        print(f"\n>>> 正在训练: {algo_name}")
        try:
            algo_cls = ALGORITHM_REGISTRY[algo_name]
            model = algo_cls(config=algo_configs.get(algo_name, None))
            model.fit(X_train, y_train, X_val, y_val)
            model_dict[algo_name] = model
            print(f"{algo_name} 训练完成")
        except Exception as e:
            print(f"{algo_name} 训练失败: {e}")
            continue

    if not model_dict:
        raise RuntimeError("所有指定算法训练均失败，程序终止")

    # ---------- 5. 批量评估与指标保存 ----------
    print("\n" + "=" * 60)
    print("【4/6】批量评估模型并保存指标")
    print("=" * 60)

    calculator = MetricsCalculator()
    try:
        metrics_result = calculator.batch_evaluate_models(
            model_dict=model_dict,
            X_test=X_test,
            y_test=y_test,
            measure_speed=config["evaluation"]["measure_inference_speed"]
        )
    except Exception as e:
        raise RuntimeError(f"模型评估失败: {e}")

    # 打印所有可用指标字段（方便排查速度字段）
    first_algo = list(metrics_result.keys())[0]
    print(f"\n📌 指标字段列表：{list(metrics_result[first_algo].keys())}")

    metrics_df = pd.DataFrame(metrics_result).T
    metrics_df.index.name = "algorithm"
    metrics_csv_path = os.path.join(run_output_dir, "metrics_comparison.csv")
    metrics_df.to_csv(metrics_csv_path, encoding="utf-8-sig", index=True)

    print("\n测试集核心指标汇总：")
    core_metrics = ["accuracy", "precision", "recall", "f1"]
    core_metrics = [m for m in core_metrics if m in metrics_df.columns]
    print(metrics_df[core_metrics].round(4))
    print(f"\n指标表格已保存: {Path(metrics_csv_path).absolute()}")

    # ---------- 6. 深度分析 ----------
    noise_results = {}
    if not args.no_analysis:
        print("\n" + "=" * 60)
        print("【5/6】执行深度分析（过拟合 + 鲁棒性）")
        print("=" * 60)

        analyzer = ModelAnalyzer()
        report_lines = []
        
        report_lines.append("# DryBean 算法对比实验 深度分析报告")
        report_lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据集类型：Clean 干净数据集")
        report_lines.append(f"核心评估指标：{main_metric}")
        report_lines.append(f"运行算法：{', '.join(model_dict.keys())}")
        report_lines.append("")

        # 6.1 过拟合分析
        report_lines.append("## 1. 过拟合分析")
        for name, model in model_dict.items():
            try:
                res = analyzer.analyze_overfitting(model, X_train, y_train, X_test, y_test, main_metric)
                report_lines.append(f"### {name}")
                report_lines.append(f"- 训练集{main_metric}：{res[f'train_{main_metric}']:.4f}")
                report_lines.append(f"- 测试集{main_metric}：{res[f'test_{main_metric}']:.4f}")
                report_lines.append(f"- 性能差距：{res['gap']:.4f}")
                report_lines.append(f"- 拟合等级：**{res['overfitting_level']}**")
                report_lines.append(f"- 优化建议：{res['suggestion']}")
                report_lines.append("")
                print(f"{name} 过拟合分析完成：{res['overfitting_level']}")
            except Exception as e:
                print(f"{name} 过拟合分析失败: {e}")
                report_lines.append(f"### {name}\n- 分析失败：{e}\n")

        # 6.2 噪声鲁棒性分析
        report_lines.append("## 2. 噪声鲁棒性分析")
        for name, model in model_dict.items():
            try:
                res = analyzer.analyze_noise_robustness(model, X_test, y_test, noise_ratios, main_metric)
                noise_results[name] = res.get("scores", res.get("noise_scores", []))
                report_lines.append(f"### {name}")
                report_lines.append(f"- 原始{main_metric}：{res['original_score']:.4f}")
                report_lines.append(f"- 鲁棒性等级：**{res['robustness_level']}**")
                report_lines.append(f"- 结论：{res['conclusion']}")
                report_lines.append("")
                print(f"{name} 鲁棒性分析完成：{res['robustness_level']}")
            except Exception as e:
                print(f"{name} 鲁棒性分析失败: {e}")
                report_lines.append(f"### {name}\n- 分析失败：{e}\n")
                noise_results[name] = []

        # 打印鲁棒性数据概况（方便排查09图问题）
        print(f"\n📌 鲁棒性数据概况：")
        for algo, scores in noise_results.items():
            print(f"  - {algo}: {len(scores)} 个数据点")

        report_path = os.path.join(run_output_dir, "analysis_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        print(f"\n深度分析报告已保存: {Path(report_path).absolute()}")

    # ---------- 7. 可视化绘图 ----------
    if not args.no_plot:
        print("\n" + "=" * 60)
        print("【6/6】生成可视化图表")
        print("=" * 60)

        viz_cfg = config["visualization"]
        plotter = ModelPlotter(
            style=viz_cfg["style"],
            dpi=viz_cfg["dpi"],
            figsize=(viz_cfg["figsize_width"], viz_cfg["figsize_height"])
        )

        # 7.1 多算法精度指标对比图
        try:
            compare_fig_path = os.path.join(plot_dir, "01_metrics_comparison.png")
            plotter.plot_metrics_comparison(
                metrics_dict=metrics_result,
                metrics=["accuracy", "precision", "recall", "f1"],
                title="多算法分类指标对比 (Clean数据集)",
                save_path=compare_fig_path
            )
            print(f"01 指标对比图已生成: {compare_fig_path}")
        except Exception as e:
            print(f"❌ 指标对比图生成失败: {e}")

        # 7.2 算法推理速度对比图（修复：显式指定速度字段）
        try:
            speed_fig_path = os.path.join(plot_dir, "02_speed_comparison.png")
            # 显式指定速度字段（avg_time_per_sample_ms 或 throughput_samples_per_sec）
            speed_success = plotter.plot_speed_comparison(
                metrics_dict=metrics_result,
                speed_field="avg_time_per_sample_ms",  # 新增：指定要读取的速度字段
                title="单样本推理耗时对比 (毫秒，越小越好)",
                save_path=speed_fig_path
            )
            if speed_success:
                print(f"02 推理速度对比图已生成: {speed_fig_path}")
        except Exception as e:
            print(f"❌ 推理速度对比图生成失败: {e}")
        
        # 7.3 各算法混淆矩阵
        for name, model in model_dict.items():
            try:
                y_pred = model.predict(X_test)
                cm_fig_path = os.path.join(plot_dir, f"03_cm_{name}.png")
                plotter.plot_confusion_matrix(
                    y_true=y_test, y_pred=y_pred,
                    class_names=CLASS_NAMES,
                    title=f"{name} 混淆矩阵",
                    save_path=cm_fig_path
                )
                print(f"03 {name} 混淆矩阵已生成: {cm_fig_path}")
            except Exception as e:
                print(f"❌ {name} 混淆矩阵生成失败: {e}")

        # 7.4 单算法损失曲线 + 收集联合对比数据
        loss_dict = {}
        for name, model in model_dict.items():
            try:
                if hasattr(model, "get_loss_curves"):
                    train_loss, val_loss = model.get_loss_curves()
                    if train_loss and len(train_loss) > 0:
                        loss_dict[name] = train_loss
                        loss_fig_path = os.path.join(plot_dir, f"04_loss_{name}.png")
                        plotter.plot_loss_curve(
                            loss_history=train_loss,
                            val_loss_history=val_loss,
                            title=f"{name} 训练-验证损失曲线",
                            save_path=loss_fig_path
                        )
                        print(f"04 {name} 损失曲线已生成: {loss_fig_path}")
            except Exception as e:
                print(f"❌ {name} 损失曲线生成失败: {e}")

        # 7.5 多算法联合Loss曲线对比图
        try:
            if len(loss_dict) >= 2:
                joint_loss_path = os.path.join(plot_dir, "05_joint_loss_comparison.png")
                plotter.plot_joint_loss_curve(
                    loss_dict=loss_dict,
                    title="多算法训练损失曲线对比",
                    save_path=joint_loss_path
                )
                print(f"05 联合损失对比图已生成: {joint_loss_path}")
        except Exception as e:
            print(f"❌ 联合损失对比图生成失败: {e}")

        # 7.6 各算法多分类ROC曲线
        for name, model in model_dict.items():
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)
                    roc_fig_path = os.path.join(plot_dir, f"06_roc_{name}.png")
                    plotter.plot_multiclass_roc(
                        y_true=y_test, y_proba=y_proba,
                        class_names=CLASS_NAMES,
                        title=f"{name} 多分类ROC曲线",
                        save_path=roc_fig_path
                    )
                    print(f"06 {name} ROC曲线已生成: {roc_fig_path}")
            except Exception as e:
                print(f"❌ {name} ROC曲线生成失败: {e}")

        # 7.7 XGBoost特征重要性图
        try:
            if "xgboost" in model_dict:
                fi_fig_path = os.path.join(plot_dir, "07_xgboost_feature_importance.png")
                xgb_model = model_dict["xgboost"]
                importance = xgb_model.get_feature_importance()
                plotter.plot_feature_importance(
                    importance=importance,
                    feature_names=feature_names,
                    title="XGBoost 特征重要性排序",
                    save_path=fi_fig_path
                )
                print(f"07 XGBoost特征重要性图已生成: {fi_fig_path}")
        except Exception as e:
            print(f"❌ XGBoost特征重要性图生成失败: {e}")

        # 7.8 各算法学习曲线
        for name, model in model_dict.items():
            try:
                if hasattr(model, "get_loss_curves"):
                    train_loss, val_loss = model.get_loss_curves()
                    if train_loss and len(train_loss) > 0:
                        train_scores = [-x for x in train_loss]
                        val_scores = [-x for x in val_loss] if (val_loss and len(val_loss) > 0) else None
                        lc_fig_path = os.path.join(plot_dir, f"08_learning_curve_{name}.png")
                        plotter.plot_learning_curve(
                            train_scores=train_scores,
                            val_scores=val_scores,
                            metric_name="负对数损失（越大越好）",
                            title=f"{name} 学习曲线",
                            save_path=lc_fig_path
                        )
                        print(f"08 {name} 学习曲线图已生成: {lc_fig_path}")
            except Exception as e:
                print(f"❌ {name} 学习曲线图生成失败: {e}")

        # 7.9 噪声鲁棒性折线对比图（修复：统一变量+明确返回值）
        try:
            if not args.no_analysis and noise_results:
                noise_fig_path = os.path.join(plot_dir, "09_noise_robustness.png")
                noise_success = plotter.plot_noise_robustness(
                    noise_results=noise_results,
                    noise_ratios=noise_ratios,
                    title="算法噪声鲁棒性对比",
                    metric_name=main_metric,
                    save_path=noise_fig_path
                )
                if noise_success:
                    print(f"09 噪声鲁棒性对比图已生成: {noise_fig_path}")
        except Exception as e:
            print(f"❌ 噪声鲁棒性对比图生成失败: {e}")

    # ---------- 8. 保存模型 ----------
    if args.save_model:
        model_save_dir = config["paths"]["model_save_dir"]
        ensure_dir(model_save_dir)
        print("\n" + "=" * 60)
        print("正在持久化模型文件...")
        print("=" * 60)
        
        for name, model in model_dict.items():
            try:
                model_path = os.path.join(model_save_dir, f"{name}_{timestamp}.pkl")
                model.save_model(model_path)
                print(f"{name} 模型已保存: {Path(model_path).absolute()}")
            except Exception as e:
                print(f"{name} 模型保存失败: {e}")

    # ---------- 结束 ----------
    print("\n" + "=" * 60)
    print("实验全部执行完成！")
    print(f"所有结果已输出到: {Path(run_output_dir).absolute()}")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断运行")
        sys.exit(1)
    except Exception as e:
        print(f"\n程序执行出错: {e}")
        sys.exit(1)