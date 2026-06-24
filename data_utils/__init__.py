# 数据模块全局常量
DATASET_ROOT_DIR = "./DryBeanDataset"  # 数据集根目录，匹配你的路径
TRAIN_CSV_PATH = f"{DATASET_ROOT_DIR}/Dry_Bean_Dataset_Clean_train.csv"
VAL_CSV_PATH = f"{DATASET_ROOT_DIR}/Dry_Bean_Dataset_Clean_val.csv"
TEST_CSV_PATH = f"{DATASET_ROOT_DIR}/Dry_Bean_Dataset_Clean_test.csv"
LABEL_COLUMN = "Class"  # 标签列名，匹配你的数据集

# 核心功能导出
from .loader import DataLoader
from .preprocess import DataPreprocessor
from .noise import NoiseInjector

__all__ = [
    "DataLoader",
    "Preprocessor",
    "NoiseInjector",
    "TRAIN_CSV_PATH",
    "VAL_CSV_PATH",
    "TEST_CSV_PATH",
    "LABEL_COLUMN"
]