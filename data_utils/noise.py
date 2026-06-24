import numpy as np
import pandas as pd
from typing import Tuple, Union
from . import LABEL_COLUMN


class NoiseInjector:
    """
    噪声注入器，用于算法鲁棒性测试
    支持3种噪声类型：高斯噪声（特征）、椒盐噪声（特征）、标签翻转（标签）
    规范：仅对训练集加噪声，验证/测试集保持原始干净数据
    """
    def __init__(self, random_seed: int = 42):
        """
        初始化噪声注入器
        :param random_seed: 随机种子，保证噪声可复现
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def add_gaussian_noise(self, X: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
        """
        给特征添加高斯噪声
        :param X: 原始特征DataFrame
        :param noise_level: 噪声强度，0~1之间，值越大噪声越强
        :return: 添加噪声后的特征DataFrame
        """
        if not 0 <= noise_level <= 1:
            raise ValueError("噪声强度noise_level必须在0~1之间")
        # 计算每个特征的标准差，按比例添加高斯噪声
        X_noisy = X.copy()
        for col in X.columns:
            col_std = X[col].std()
            # 高斯噪声：均值0，标准差=噪声强度*特征标准差
            noise = np.random.normal(loc=0, scale=noise_level * col_std, size=len(X))
            X_noisy[col] = X[col] + noise
        return X_noisy

    def add_salt_pepper_noise(self, X: pd.DataFrame, noise_level: float = 0.1) -> pd.DataFrame:
        """
        给特征添加椒盐噪声（随机将部分值替换为该列的最大/最小值）
        :param X: 原始特征DataFrame
        :param noise_level: 噪声强度，0~1之间，值越大噪声越强
        :return: 添加噪声后的特征DataFrame
        """
        if not 0 <= noise_level <= 1:
            raise ValueError("噪声强度noise_level必须在0~1之间")
        X_noisy = X.copy()
        n_samples, n_features = X.shape
        # 计算需要加噪声的元素总数
        total_noise_elements = int(n_samples * n_features * noise_level)
        # 随机选择要加噪声的位置
        noise_indices = np.random.choice(n_samples * n_features, total_noise_elements, replace=False)
        # 展平数组，方便批量修改
        X_flat = X_noisy.values.flatten()
        # 计算每个特征的最大最小值
        col_min = X.min().values
        col_max = X.max().values
        # 给选中的位置添加椒盐噪声
        for idx in noise_indices:
            col_idx = idx % n_features
            # 50%概率替换为最小值（椒），50%替换为最大值（盐）
            if np.random.random() < 0.5:
                X_flat[idx] = col_min[col_idx]
            else:
                X_flat[idx] = col_max[col_idx]
        # 恢复二维形状
        X_noisy = pd.DataFrame(X_flat.reshape(n_samples, n_features), columns=X.columns, index=X.index)
        return X_noisy

    def flip_labels(self, y: pd.Series, flip_level: float = 0.1) -> pd.Series:
        """
        给标签添加翻转噪声（随机将部分样本的标签替换为其他类别）
        :param y: 原始标签Series
        :param flip_level: 翻转强度，0~1之间，值越大翻转的样本越多
        :return: 翻转后的标签Series
        """
        if not 0 <= flip_level <= 1:
            raise ValueError("标签翻转强度flip_level必须在0~1之间")
        y_flipped = y.copy()
        n_samples = len(y)
        # 计算需要翻转的样本数
        n_flip = int(n_samples * flip_level)
        # 随机选择要翻转的样本索引
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        # 获取所有唯一标签
        unique_labels = y.unique()
        # 翻转选中的样本标签
        for idx in flip_indices:
            original_label = y.iloc[idx]
            # 随机选择一个不同于原始标签的新标签
            new_label = np.random.choice([l for l in unique_labels if l != original_label])
            y_flipped.iloc[idx] = new_label
        return y_flipped

    def add_noise(self, 
                  X: pd.DataFrame, 
                  y: pd.Series, 
                  noise_type: str = "gaussian", 
                  noise_level: float = 0.1) -> Tuple[pd.DataFrame, pd.Series]:
        """
        一站式噪声添加方法，支持所有噪声类型
        :param X: 原始特征
        :param y: 原始标签
        :param noise_type: 噪声类型，可选gaussian/salt_pepper/label_flip
        :param noise_level: 噪声强度，0~1之间
        :return: 添加噪声后的特征、标签
        """
        if noise_type == "gaussian":
            X_noisy = self.add_gaussian_noise(X, noise_level)
            y_noisy = y.copy()
        elif noise_type == "salt_pepper":
            X_noisy = self.add_salt_pepper_noise(X, noise_level)
            y_noisy = y.copy()
        elif noise_type == "label_flip":
            X_noisy = X.copy()
            y_noisy = self.flip_labels(y, noise_level)
        else:
            raise ValueError(f"不支持的噪声类型：{noise_type}，可选gaussian/salt_pepper/label_flip")
        return X_noisy, y_noisy