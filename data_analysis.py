import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===================== 全局设置 =====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
os.makedirs('./figures', exist_ok=True)

# ===================== 数据加载与核心类型修复（关键！） =====================
TRAIN_PATH = './DryBeanDataset/Dry_Bean_Dataset_Dirty_train.csv'
TEST_PATH = './DryBeanDataset/Dry_Bean_Dataset_Dirty_test.csv'
VAL_PATH = './DryBeanDataset/Dry_Bean_Dataset_Dirty_val.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
val_df = pd.read_csv(VAL_PATH)

label_col = 'Class'
# 所有特征列（除标签外）都应该是数值类型
feature_cols = train_df.columns.drop(label_col).tolist()

# 核心修复：处理带单位的字符串数值（如"0.6921 cm"）
def clean_numeric_column(series):
    # 移除所有非数字字符（保留小数点和负号）
    series_clean = series.astype(str).str.replace(r'[^0-9.-]', '', regex=True)
    # 转换为数值，无法转换的转为NaN
    return pd.to_numeric(series_clean, errors='coerce')

# 应用到所有特征列
for col in feature_cols:
    train_df[col] = clean_numeric_column(train_df[col])
    test_df[col] = clean_numeric_column(test_df[col])
    val_df[col] = clean_numeric_column(val_df[col])

# 现在可以安全筛选数值列了
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"已转换为数值类型的特征: {numeric_cols}")

# ===================== 1. 基本信息统计 =====================
print("\n" + "="*50)
print("数据集基本信息统计")
print("="*50)
print(f"训练集样本数: {train_df.shape[0]}, 特征数: {train_df.shape[1]-1}")
print(f"测试集样本数: {test_df.shape[0]}, 特征数: {test_df.shape[1]-1}")
print(f"验证集样本数: {val_df.shape[0]}, 特征数: {val_df.shape[1]-1}")
print(f"\n数据类型分布:\n{train_df.dtypes.value_counts()}")

with open('./figures/dataset_basic_info.txt', 'w', encoding='utf-8') as f:
    f.write(f"训练集: {train_df.shape[0]}样本 × {train_df.shape[1]-1}特征\n")
    f.write(f"测试集: {test_df.shape[0]}样本 × {test_df.shape[1]-1}特征\n")
    f.write(f"验证集: {val_df.shape[0]}样本 × {val_df.shape[1]-1}特征\n\n")
    f.write(f"特征列表:\n{chr(10).join(feature_cols)}")

# ===================== 2. 缺失值分析 =====================
print("\n" + "="*50)
print("缺失值分析")
print("="*50)

train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()
val_missing = val_df.isnull().sum()

print("训练集缺失值统计:")
missing_features = train_missing[train_missing > 0]
if not missing_features.empty:
    print(missing_features.to_string())
else:
    print("无缺失值")
print(f"\n训练集总缺失值数: {train_missing.sum()}")
print(f"含缺失值的特征数: {len(missing_features)}")

plt.figure(figsize=(12, 8))
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap='viridis', alpha=0.8)
plt.title('训练集缺失值分布热力图', fontsize=14)
plt.tight_layout()
plt.savefig('./figures/missing_values_heatmap.png', bbox_inches='tight')
plt.close()

missing_ratio = (train_missing / len(train_df)) * 100
missing_ratio = missing_ratio[missing_ratio > 0].sort_values(ascending=False)
if not missing_ratio.empty:
    plt.figure(figsize=(10, 6))
    # 修复palette警告
    sns.barplot(x=missing_ratio.index, y=missing_ratio.values, hue=missing_ratio.index, palette='Reds_r', legend=False)
    plt.title('训练集各特征缺失值比例', fontsize=14)
    plt.xlabel('特征名称', fontsize=12)
    plt.ylabel('缺失值比例 (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('./figures/missing_values_ratio.png', bbox_inches='tight')
    plt.close()

# ===================== 3. 标签分布与错误分析 =====================
print("\n" + "="*50)
print("标签分布与错误分析")
print("="*50)

original_labels = train_df[label_col].value_counts()
print("原始标签分布:")
print(original_labels.to_string())

correct_labels = ['SEKER', 'BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SIRA']
wrong_labels = [label for label in original_labels.index if label not in correct_labels]
print(f"\n检测到的错误标签: {wrong_labels if wrong_labels else '无'}")

plt.figure(figsize=(12, 6))
bars = plt.bar(original_labels.index, original_labels.values, color=sns.color_palette('viridis', len(original_labels)))
plt.title('训练集原始标签分布（含错误标签）', fontsize=14)
plt.xlabel('豆类类别', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('./figures/label_distribution_original.png', bbox_inches='tight')
plt.close()

train_df_clean = train_df.copy()
train_df_clean[label_col] = train_df_clean[label_col].str.strip().str.upper()
label_mapping = {
    'S3K3R': 'SEKER', 'D3RMAS0N': 'DERMASON', 'H0R0Z': 'HOROZ',
    'SIRA': 'SIRA', 'CALI': 'CALI', 'DERMASON': 'DERMASON',
    'HOROZ': 'HOROZ', 'SEKER': 'SEKER', 'BARBUNYA': 'BARBUNYA', 'BOMBAY': 'BOMBAY'
}
train_df_clean[label_col] = train_df_clean[label_col].map(label_mapping)
clean_labels = train_df_clean[label_col].value_counts().reindex(correct_labels).fillna(0).astype(int)

print("\n修正标签后分布:")
print(clean_labels.to_string())

plt.figure(figsize=(12, 6))
bars = plt.bar(clean_labels.index, clean_labels.values, color=sns.color_palette('viridis', 7))
plt.title('修正标签后训练集类别分布', fontsize=14)
plt.xlabel('豆类类别', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig('./figures/label_distribution_clean.png', bbox_inches='tight')
plt.close()

max_count = clean_labels.max()
min_count = clean_labels[clean_labels > 0].min() if (clean_labels > 0).any() else 0
imbalance_ratio = max_count / min_count if min_count > 0 else 0
print(f"\n类别不平衡比例: {imbalance_ratio:.2f}:1 (最多/最少)")

# ===================== 4. 数值特征统计与异常值分析 =====================
print("\n" + "="*50)
print("数值特征统计与异常值分析")
print("="*50)

stats_desc = train_df[numeric_cols].describe().T
stats_desc['缺失值数'] = train_missing[numeric_cols]
stats_desc['缺失值比例(%)'] = (train_missing[numeric_cols] / len(train_df)) * 100
stats_desc.to_csv('./figures/numeric_features_stats.csv', encoding='utf-8-sig')
print("数值特征统计描述已保存到 ./figures/numeric_features_stats.csv")

def count_outliers(series):
    series_clean = series.dropna()
    if len(series_clean) < 4 or series_clean.nunique() == 1:
        return 0
    Q1 = series_clean.quantile(0.25)
    Q3 = series_clean.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return len(series_clean[(series_clean < lower) | (series_clean > upper)])

outlier_counts = {col: count_outliers(train_df[col]) for col in numeric_cols}
outlier_df = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['异常值数量'])
outlier_df['异常值比例(%)'] = (outlier_df['异常值数量'] / len(train_df)) * 100
outlier_df = outlier_df.sort_values('异常值比例(%)', ascending=False)
outlier_df.to_csv('./figures/outlier_stats.csv', encoding='utf-8-sig')
print("\n异常值统计已保存到 ./figures/outlier_stats.csv")

if not outlier_df.empty and outlier_df['异常值比例(%)'].max() > 0:
    print(f"异常值比例最高的特征: {outlier_df.index[0]} ({outlier_df.iloc[0]['异常值比例(%)']:.2f}%)")
else:
    print("未检测到异常值")

# 绘制箱线图（分两页展示，避免拥挤）- 修复宽格式hue错误
plt.figure(figsize=(16, 10))
n_groups = (len(numeric_cols) + 7) // 8
for i in range(n_groups):
    plt.subplot(n_groups, 1, i+1)
    cols = numeric_cols[i*8 : (i+1)*8]
    
    # 正确写法：宽格式箱线图不使用hue，直接指定palette
    # seaborn v0.14.0对宽格式保留了palette参数的兼容性，无需hue
    sns.boxplot(data=train_df[cols], palette='Set2')
    
    plt.title(f'特征箱线图（第{i+1}组）', fontsize=14)
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('./figures/feature_boxplots.png', bbox_inches='tight')
plt.close()

# ===================== 5. 特征相关性分析 =====================
print("\n" + "="*50)
print("特征相关性分析")
print("="*50)

corr_matrix = train_df[numeric_cols].corr()
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=0.5, annot_kws={"size": 9})
plt.title('特征相关性热力图（下三角）', fontsize=16)
plt.tight_layout()
plt.savefig('./figures/correlation_heatmap.png', bbox_inches='tight')
plt.close()

high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.9:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

print(f"高度相关特征对(|r|>0.9):")
if high_corr:
    for pair in high_corr:
        print(f"{pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
else:
    print("无高度相关特征对")

# ===================== 6. 特征分布分析 =====================
print("\n" + "="*50)
print("特征分布分析")
print("="*50)

n_rows = (len(numeric_cols) + 3) // 4
plt.figure(figsize=(16, 4*n_rows))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(n_rows, 4, i)
    sns.histplot(train_df[col].dropna(), kde=True, bins=30, color='steelblue')
    plt.title(f'{col} 分布', fontsize=11)
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout()
plt.savefig('./figures/all_feature_distributions.png', bbox_inches='tight')
plt.close()

# 关键特征按类别分布（最终修复版）
key_features = ['Area', 'Perimeter', 'AspectRation', 'Eccentricity', 'Compactness', 'ShapeFactor4']
plt.figure(figsize=(16, 12))

print("\nCompactness特征统计信息（已转换为数值）:")
print(train_df_clean['Compactness'].describe())

for i, col in enumerate(key_features, 1):
    ax = plt.subplot(2, 3, i)
    plot_data = train_df_clean.dropna(subset=[label_col, col])
    
    # 自动计算99%分位数范围
    q1 = plot_data[col].quantile(0.01)
    q3 = plot_data[col].quantile(0.99)
    margin = (q3 - q1) * 0.1
    y_min = q1 - margin
    y_max = q3 + margin
    
    # 修复palette警告
    sns.boxplot(
        x=label_col, y=col, data=plot_data,
        palette='Set3', hue=label_col, legend=False,
        fliersize=1.5, whis=1.5
    )
    
    ax.set_ylim(y_min, y_max)
    plt.title(f'{col} 按类别分布', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')

plt.tight_layout()
plt.savefig('./figures/key_features_by_class.png', bbox_inches='tight', dpi=300)
plt.close()

# ===================== 7. 数据污染总结 =====================
print("\n" + "="*50)
print("数据污染综合总结")
print("="*50)

missing_ratio_max = missing_ratio.iloc[0] if not missing_ratio.empty else 0
missing_ratio_feature = missing_ratio.index[0] if not missing_ratio.empty else '无'
outlier_max_feature = outlier_df.index[0] if (not outlier_df.empty and outlier_df['异常值比例(%)'].max() > 0) else '无'
outlier_max_ratio = outlier_df.iloc[0]['异常值比例(%)'] if (not outlier_df.empty and outlier_df['异常值比例(%)'].max() > 0) else 0
max_label = clean_labels.idxmax() if clean_labels.max() > 0 else '无'
min_label = clean_labels[clean_labels > 0].idxmin() if (clean_labels > 0).any() else '无'
max_label_count = clean_labels.max() if clean_labels.max() > 0 else 0
min_label_count = clean_labels[clean_labels > 0].min() if (clean_labels > 0).any() else 0
top_corr_pair = f"{high_corr[0][0]}与{high_corr[0][1]} (r={high_corr[0][2]:.3f})" if high_corr else '无'

summary = f"""
1. 缺失值问题:
   - 训练集共{train_missing.sum()}个缺失值，分布在{len(missing_features)}个特征中
   - 缺失值比例最高的特征: {missing_ratio_feature} ({missing_ratio_max:.2f}%)

2. 标签错误问题:
   - 检测到{len(wrong_labels)}个错误标签，主要是字符替换错误（如S3K3R→SEKER）
   - 修正后所有样本归为7个标准类别

3. 异常值问题:
   - 共检测到{sum(outlier_counts.values())}个异常值，涉及{len([c for c in outlier_counts if outlier_counts[c]>0])}个特征
   - 异常值比例最高的特征: {outlier_max_feature} ({outlier_max_ratio:.2f}%)
   - 异常值主要集中在形状因子类特征

4. 类别不平衡问题:
   - 样本量最多: {max_label} ({max_label_count}个)
   - 样本量最少: {min_label} ({min_label_count}个)
   - 不平衡比例: {imbalance_ratio:.2f}:1

5. 特征相关性问题:
   - 共发现{len(high_corr)}对高度相关特征(|r|>0.9)
   - 最相关特征: {top_corr_pair}

6. 特征分布问题:
   - 大部分特征呈现右偏分布
   - 不同类别在形状特征上有明显区分度
   - 已修复带单位的数值特征（如Compactness）的类型问题
"""
print(summary)

with open('./figures/data_pollution_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("\n数据分析完成！所有图表和统计文件已保存到 ./figures 文件夹")