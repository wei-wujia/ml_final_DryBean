import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# ===================== 全局设置 =====================
# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# 路径设置（完全匹配项目结构）
RAW_DATA_DIR = './DryBeanDataset'
FIGURES_DIR = './figures_processed'
os.makedirs(FIGURES_DIR, exist_ok=True)

# 原始数据路径
TRAIN_RAW_PATH = os.path.join(RAW_DATA_DIR, 'Dry_Bean_Dataset_Dirty_train.csv')
VAL_RAW_PATH = os.path.join(RAW_DATA_DIR, 'Dry_Bean_Dataset_Dirty_val.csv')
TEST_RAW_PATH = os.path.join(RAW_DATA_DIR, 'Dry_Bean_Dataset_Dirty_test.csv')

# 清洗后数据保存路径
TRAIN_CLEAN_PATH = os.path.join(RAW_DATA_DIR, 'Dry_Bean_Dataset_Clean_train.csv')
VAL_CLEAN_PATH = os.path.join(RAW_DATA_DIR, 'Dry_Bean_Dataset_Clean_val.csv')
TEST_CLEAN_PATH = os.path.join(RAW_DATA_DIR, 'Dry_Bean_Dataset_Clean_test.csv')

# 全局常量
LABEL_COL = 'Class'
CORR_THRESHOLD = 0.9  # 相关性阈值，超过则剔除冗余特征
RANDOM_SEED = 42  # 随机种子，保证结果可复现

# ===================== 工具函数 =====================
def clean_numeric_string(s):
    """清洗数值字符串，去除所有非数字/小数点/负号的字符"""
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    # 正则提取数字部分
    num_match = re.search(r'[-+]?\d+\.?\d*', s)
    if num_match:
        return float(num_match.group())
    else:
        return np.nan

# ===================== 1. 数据加载 =====================
print("="*60)
print("1. 数据加载")
print("="*60)

# 加载原始数据
train_df = pd.read_csv(TRAIN_RAW_PATH)
val_df = pd.read_csv(VAL_RAW_PATH)
test_df = pd.read_csv(TEST_RAW_PATH)

print(f"原始训练集形状: {train_df.shape}")
print(f"原始验证集形状: {val_df.shape}")
print(f"原始测试集形状: {test_df.shape}")

# 分离特征列和标签列
feature_cols = train_df.columns.drop(LABEL_COL).tolist()
print(f"原始特征列: {feature_cols}")

# ===================== 2. 数据清洗 =====================
print("\n" + "="*60)
print("2. 数据清洗")
print("="*60)

# 2.1 数值特征清洗：去除单位、强制转换为数值类型
print("\n2.1 数值特征清洗")
for col in feature_cols:
    # 对所有特征列应用清洗函数
    train_df[col] = train_df[col].apply(clean_numeric_string)
    val_df[col] = val_df[col].apply(clean_numeric_string)
    test_df[col] = test_df[col].apply(clean_numeric_string)
    
    # 转换为float类型
    train_df[col] = train_df[col].astype(float)
    val_df[col] = val_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)

print("数值特征清洗完成，所有特征已转换为float类型")

# 2.2 标签清洗：修正错误标签
print("\n2.2 标签清洗")
# 标准标签列表
correct_labels = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SIRA']
# 标签修正映射表（覆盖所有错误类型）
label_mapping = {
    # 字符替换错误
    'S3K3R': 'SEKER', 'D3RMAS0N': 'DERMASON', 'H0R0Z': 'HOROZ', 'B0MBAY': 'BOMBAY',
    # 大小写错误
    'dermason': 'DERMASON', 'sira': 'SIRA', 'horoz': 'HOROZ', 'seker': 'SEKER',
    'barbunya': 'BARBUNYA', 'cali': 'CALI', 'bombay': 'BOMBAY',
    # 前后空格错误
    'DERMASON ': 'DERMASON', 'SIRA ': 'SIRA', 'HOROZ ': 'HOROZ', 'SEKER ': 'SEKER',
    'BARBUNYA ': 'BARBUNYA', 'CALI ': 'CALI', 'BOMBAY ': 'BOMBAY',
    # 标准标签自映射
    'SEKER': 'SEKER', 'BARBUNYA': 'BARBUNYA', 'BOMBAY': 'BOMBAY',
    'CALI': 'CALI', 'DERMASON': 'DERMASON', 'HOROZ': 'HOROZ', 'SIRA': 'SIRA'
}

# 应用标签修正
def clean_label(label):
    if pd.isna(label):
        return np.nan
    label = str(label).strip().upper()
    return label_mapping.get(label, np.nan)

train_df[LABEL_COL] = train_df[LABEL_COL].apply(clean_label)
val_df[LABEL_COL] = val_df[LABEL_COL].apply(clean_label)
test_df[LABEL_COL] = test_df[LABEL_COL].apply(clean_label)

# 剔除标签为NaN的样本
train_df = train_df.dropna(subset=[LABEL_COL])
val_df = val_df.dropna(subset=[LABEL_COL])
test_df = test_df.dropna(subset=[LABEL_COL])

print(f"标签清洗完成，训练集剩余样本数: {len(train_df)}")
print(f"标签清洗完成，验证集剩余样本数: {len(val_df)}")
print(f"标签清洗完成，测试集剩余样本数: {len(test_df)}")

# 2.3 缺失值填充：使用训练集均值填充，避免数据泄露
print("\n2.3 缺失值填充")
# 计算训练集各特征的均值
train_mean = train_df[feature_cols].mean()
print(f"训练集各特征均值:\n{train_mean}")

# 填充缺失值
train_df[feature_cols] = train_df[feature_cols].fillna(train_mean)
val_df[feature_cols] = val_df[feature_cols].fillna(train_mean)
test_df[feature_cols] = test_df[feature_cols].fillna(train_mean)

print(f"缺失值填充完成，训练集剩余缺失值数: {train_df.isnull().sum().sum()}")
print(f"缺失值填充完成，验证集剩余缺失值数: {val_df.isnull().sum().sum()}")
print(f"缺失值填充完成，测试集剩余缺失值数: {test_df.isnull().sum().sum()}")

# 2.4 异常值处理：1%~99%分位数截断，使用训练集分位数
print("\n2.4 异常值处理")
# 计算训练集各特征的1%和99%分位数
train_quantile_1 = train_df[feature_cols].quantile(0.01)
train_quantile_99 = train_df[feature_cols].quantile(0.99)
print(f"训练集各特征1%分位数:\n{train_quantile_1}")
print(f"训练集各特征99%分位数:\n{train_quantile_99}")

# 截断异常值
for col in feature_cols:
    train_df[col] = np.clip(train_df[col], train_quantile_1[col], train_quantile_99[col])
    val_df[col] = np.clip(val_df[col], train_quantile_1[col], train_quantile_99[col])
    test_df[col] = np.clip(test_df[col], train_quantile_1[col], train_quantile_99[col])

print("异常值截断完成，所有特征值已限制在1%~99%分位数范围内")

# ===================== 3. 特征工程 =====================
print("\n" + "="*60)
print("3. 特征工程")
print("="*60)

# 3.1 特征选择1：剔除低方差特征
print("\n3.1 低方差特征剔除")
var_selector = VarianceThreshold(threshold=0.001)
var_selector.fit(train_df[feature_cols])
selected_feature_cols = var_selector.get_feature_names_out().tolist()
print(f"低方差特征剔除后，剩余特征数: {len(selected_feature_cols)}")
print(f"剩余特征列表: {selected_feature_cols}")

# 3.2 特征选择2：剔除高度共线性特征
print("\n3.2 高度共线性特征剔除")
# 计算训练集特征相关性矩阵
corr_matrix = train_df[selected_feature_cols].corr().abs()
# 生成上三角矩阵，避免重复判断
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
# 找出相关性超过阈值的特征列
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > CORR_THRESHOLD)]
print(f"高度共线性特征（|r|>{CORR_THRESHOLD}）: {to_drop}")

# 剔除冗余特征
final_feature_cols = [col for col in selected_feature_cols if col not in to_drop]
print(f"共线性剔除后，最终特征数: {len(final_feature_cols)}")
print(f"最终特征列表: {final_feature_cols}")

# 3.3 特征标准化：StandardScaler，使用训练集拟合
print("\n3.3 特征标准化")
scaler = StandardScaler()
# 拟合训练集
scaler.fit(train_df[final_feature_cols])
# 转换全量数据集
train_df[final_feature_cols] = scaler.transform(train_df[final_feature_cols])
val_df[final_feature_cols] = scaler.transform(val_df[final_feature_cols])
test_df[final_feature_cols] = scaler.transform(test_df[final_feature_cols])

print("特征标准化完成，所有特征已转换为均值0、标准差1的标准分布")

# 3.4 类别不平衡处理：SMOTE过采样，仅对训练集
print("\n3.4 类别不平衡处理（SMOTE过采样）")
# 分离训练集特征和标签
X_train = train_df[final_feature_cols]
y_train = train_df[LABEL_COL]

print(f"过采样前训练集标签分布:\n{y_train.value_counts()}")

# 应用SMOTE过采样
smote = SMOTE(random_state=RANDOM_SEED)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"过采样后训练集标签分布:\n{y_train_resampled.value_counts()}")
print(f"过采样后训练集样本数: {len(X_train_resampled)}")

# 重构过采样后的训练集DataFrame
train_df_resampled = pd.DataFrame(X_train_resampled, columns=final_feature_cols)
train_df_resampled[LABEL_COL] = y_train_resampled

# 验证集和测试集保持不变，仅保留最终特征列
val_df_final = val_df[final_feature_cols + [LABEL_COL]]
test_df_final = test_df[final_feature_cols + [LABEL_COL]]

# ===================== 4. 清洗后数据保存 =====================
print("\n" + "="*60)
print("4. 清洗后数据保存")
print("="*60)

# 保存清洗后的数据
train_df_resampled.to_csv(TRAIN_CLEAN_PATH, index=False, encoding='utf-8-sig')
val_df_final.to_csv(VAL_CLEAN_PATH, index=False, encoding='utf-8-sig')
test_df_final.to_csv(TEST_CLEAN_PATH, index=False, encoding='utf-8-sig')

print(f"清洗后训练集已保存至: {TRAIN_CLEAN_PATH}")
print(f"清洗后验证集已保存至: {VAL_CLEAN_PATH}")
print(f"清洗后测试集已保存至: {TEST_CLEAN_PATH}")

# 保存特征工程相关信息，用于后续模型训练
feature_info = {
    'final_feature_cols': final_feature_cols,
    'label_mapping': label_mapping,
    'correct_labels': correct_labels,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist()
}
import json
with open(os.path.join(RAW_DATA_DIR, 'feature_info.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_info, f, ensure_ascii=False, indent=4)
print(f"特征工程信息已保存至: {os.path.join(RAW_DATA_DIR, 'feature_info.json')}")

# ===================== 5. 可视化：处理前后对比图表 =====================
print("\n" + "="*60)
print("5. 生成处理前后对比图表")
print("="*60)

# 5.1 标签分布对比图
plt.figure(figsize=(16, 6))
# 原始标签分布
plt.subplot(1, 2, 1)
original_label_dist = train_df[LABEL_COL].value_counts()
# 修复palette警告：添加hue并设置legend=False
sns.barplot(x=original_label_dist.index, y=original_label_dist.values, 
            hue=original_label_dist.index, palette='viridis', legend=False)
plt.title('原始训练集标签分布', fontsize=14)
plt.xlabel('豆类类别', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 过采样后标签分布
plt.subplot(1, 2, 2)
resampled_label_dist = train_df_resampled[LABEL_COL].value_counts()
# 修复palette警告：添加hue并设置legend=False
sns.barplot(x=resampled_label_dist.index, y=resampled_label_dist.values, 
            hue=resampled_label_dist.index, palette='viridis', legend=False)
plt.title('过采样后训练集标签分布', fontsize=14)
plt.xlabel('豆类类别', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'preprocessing_label_distribution.png'), bbox_inches='tight')
plt.close()
print("标签分布对比图已生成")

# 5.2 缺失值处理前后对比图
# 增大figsize并调整布局，解决tight_layout警告
plt.figure(figsize=(14, 7))
# 原始缺失值分布
plt.subplot(1, 2, 1)
original_missing = pd.read_csv(TRAIN_RAW_PATH).isnull().sum()
original_missing = original_missing[original_missing > 0]
# 修复palette警告：添加hue并设置legend=False
sns.barplot(x=original_missing.index, y=original_missing.values, 
            hue=original_missing.index, palette='Reds_r', legend=False)
plt.title('原始训练集缺失值数量', fontsize=14)
plt.xlabel('特征名称', fontsize=12)
plt.ylabel('缺失值数量', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 处理后缺失值分布
plt.subplot(1, 2, 2)
clean_missing = train_df_resampled.isnull().sum()
clean_missing = clean_missing[clean_missing > 0]
if not clean_missing.empty:
    sns.barplot(x=clean_missing.index, y=clean_missing.values, 
                hue=clean_missing.index, palette='Reds_r', legend=False)
else:
    plt.bar([], [])
    plt.text(0.5, 0.5, '无缺失值', ha='center', va='center', fontsize=16)
plt.title('处理后训练集缺失值数量', fontsize=14)
plt.xlabel('特征名称', fontsize=12)
plt.ylabel('缺失值数量', fontsize=12)
plt.xticks(rotation=45, ha='right')

# 调整子图间距，解决tight_layout警告
plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15)
plt.savefig(os.path.join(FIGURES_DIR, 'preprocessing_missing_values.png'), bbox_inches='tight')
plt.close()
print("缺失值处理前后对比图已生成")

# 5.3 特征分布对比图（改用最终保留的特征：Area和roundness）
# 确保使用的特征在final_feature_cols中
plot_feature1 = 'Area'  # 最终保留的特征
plot_feature2 = 'roundness'  # 替换被剔除的Compactness，使用最终保留的特征
plt.figure(figsize=(16, 12))

# 原始特征分布 - Area
plt.subplot(2, 2, 1)
raw_train_df = pd.read_csv(TRAIN_RAW_PATH)
sns.histplot(raw_train_df[plot_feature1].dropna(), kde=True, bins=30, color='steelblue')
plt.title(f'原始{plot_feature1}特征分布', fontsize=14)
plt.xlabel(plot_feature1, fontsize=12)
plt.ylabel('样本数量', fontsize=12)

# 原始特征分布 - roundness
plt.subplot(2, 2, 2)
sns.histplot(raw_train_df[plot_feature2].dropna(), kde=True, bins=30, color='steelblue')
plt.title(f'原始{plot_feature2}特征分布', fontsize=14)
plt.xlabel(plot_feature2, fontsize=12)
plt.ylabel('样本数量', fontsize=12)

# 处理后特征分布 - Area
plt.subplot(2, 2, 3)
sns.histplot(train_df_resampled[plot_feature1], kde=True, bins=30, color='forestgreen')
plt.title(f'处理后{plot_feature1}特征分布', fontsize=14)
plt.xlabel(f'{plot_feature1}（标准化）', fontsize=12)
plt.ylabel('样本数量', fontsize=12)

# 处理后特征分布 - roundness
plt.subplot(2, 2, 4)
sns.histplot(train_df_resampled[plot_feature2], kde=True, bins=30, color='forestgreen')
plt.title(f'处理后{plot_feature2}特征分布', fontsize=14)
plt.xlabel(f'{plot_feature2}（标准化）', fontsize=12)
plt.ylabel('样本数量', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'preprocessing_feature_distribution.png'), bbox_inches='tight')
plt.close()
print("特征分布对比图已生成")

# 5.4 最终特征相关性热力图
plt.figure(figsize=(12, 10))
final_corr_matrix = train_df_resampled[final_feature_cols].corr()
mask = np.triu(np.ones_like(final_corr_matrix, dtype=bool))
sns.heatmap(final_corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, annot_kws={"size": 9})
plt.title('最终特征相关性热力图（下三角）', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'preprocessing_final_correlation.png'), bbox_inches='tight')
plt.close()
print("最终特征相关性热力图已生成")

print("\n数据处理与特征工程全部完成！")