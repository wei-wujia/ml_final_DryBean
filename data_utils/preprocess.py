# data_utils/preprocess.py
import numpy as np
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class DataPreprocessor:
    def __init__(self, scaler_type='standard', label_names=None):
        """
        初始化预处理器
        :param scaler_type: 标准化类型 standard/minmax
        :param label_names: 标签类别列表，用于固定编码顺序
        """
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.label_encoder = LabelEncoder()
        if label_names is not None:
            self.label_encoder.fit(label_names)
    
    @classmethod
    def load_from_file(cls, info_path):
        """从离线处理生成的feature_info.json加载已有scaler参数"""
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        preprocessor = cls(label_names=info['correct_labels'])
        preprocessor.scaler.mean_ = np.array(info['scaler_mean'])
        preprocessor.scaler.scale_ = np.array(info['scaler_scale'])
        return preprocessor

    def fit_transform_features(self, X_train):
        """训练集拟合并变换，用于交叉验证"""
        return self.scaler.fit_transform(X_train)
    
    def transform_features(self, X):
        """用已拟合的scaler变换数据"""
        return self.scaler.transform(X)
    
    def encode_labels(self, y):
        """字符串标签转数值"""
        return self.label_encoder.transform(y)
    
    def decode_labels(self, y_encoded):
        """数值标签转回字符串"""
        return self.label_encoder.inverse_transform(y_encoded)