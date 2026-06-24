"""数据加载工具：加载训练/验证/测试集，分离特征和标签"""
import pandas as pd

# 标签列名称（全局常量，与main.py对齐）
LABEL_COLUMN = "Class"

class DataLoader:
    def __init__(self, train_path: str, val_path: str, test_path: str, label_col: str = LABEL_COLUMN):
        """
        初始化数据加载器
        :param train_path: 训练集CSV路径
        :param val_path: 验证集CSV路径
        :param test_path: 测试集CSV路径
        :param label_col: 标签列名称
        """
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.label_col = label_col

    def _load_single(self, file_path: str) -> tuple[pd.DataFrame, pd.Series]:
        """加载单个数据集，分离特征和标签"""
        if not file_path or not pd.io.common.file_exists(file_path):
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        
        df = pd.read_csv(file_path)
        X = df.drop(columns=[self.label_col])
        y = df[self.label_col]
        return X, y

    def load_all(self) -> tuple[tuple[pd.DataFrame, pd.Series], 
                                tuple[pd.DataFrame, pd.Series], 
                                tuple[pd.DataFrame, pd.Series]]:
        """加载所有数据集并返回 (训练集, 验证集, 测试集)，每个集合为(X, y)元组"""
        train_data = self._load_single(self.train_path)
        val_data = self._load_single(self.val_path)
        test_data = self._load_single(self.test_path)
        return train_data, val_data, test_data
    
    def get_feature_names(self):
        """获取数据集特征列名（排除标签列）"""
        # 读取训练集表头，排除标签列（假设标签列名为 "Class"）
        df = pd.read_csv(self.train_path)
        return [col for col in df.columns if col != "Class"]