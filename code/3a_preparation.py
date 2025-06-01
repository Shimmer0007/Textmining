import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

# --- 定义文件路径 ---
base_data_path = r'D:\Codes\textmining\data'
raw_data_path = os.path.join(base_data_path, 'raw', 'data.csv')
processed_dir = os.path.join(base_data_path, 'processed') # 输出目录，用于保存处理好的标签

# --- 创建输出目录 (如果不存在) ---
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    print(f"目录已创建: {processed_dir}")
else:
    print(f"目录已存在: {processed_dir}")

output_labels_path = os.path.join(processed_dir, 'multilabel_targets_research_areas.npy')
output_label_classes_path = os.path.join(processed_dir, 'research_areas_classes.pkl') # 保存标签类别名称

# --- 1. 加载原始数据中的 "Research Areas" 列 ---
print(f"\n正在从 '{raw_data_path}' 加载 'Research Areas' 数据...")
try:
    df = pd.read_csv(raw_data_path)
    if 'Research Areas' not in df.columns:
        print(f"错误: CSV文件中未找到 'Research Areas' 列。可用的列: {df.columns.tolist()}")
        exit()
    
    # 提取 "Research Areas" 列，处理可能的NaN值（视为空标签列表）
    # .astype(str) 确保即使某行是数字也能被当作字符串处理，尽管这里期望的是文本
    research_areas_series = df['Research Areas'].fillna('').astype(str)
    print(f"成功加载 {len(research_areas_series)} 条 'Research Areas' 记录。")

except FileNotFoundError:
    print(f"错误: 文件 '{raw_data_path}' 未找到。")
    exit()
except Exception as e:
    print(f"加载 '{raw_data_path}' 时发生错误: {e}")
    exit()

# --- 2. 解析标签字符串为标签列表 ---
print("\n正在解析 'Research Areas' 字符串为标签列表...")
parsed_labels = []
for area_string in research_areas_series:
    if pd.isna(area_string) or not area_string.strip():
        parsed_labels.append([]) # 空字符串或NaN视为空列表
    else:
        # 按分号分割，并去除每个标签前后的空格
        labels = [label.strip() for label in area_string.split(';') if label.strip()]
        parsed_labels.append(labels)

# 打印前几个解析后的标签作为示例
print("前5个文档解析后的 'Research Areas' 标签示例:")
for i in range(min(5, len(parsed_labels))):
    print(f"  文档 {i}: {parsed_labels[i]}")

# --- 3. 使用 MultiLabelBinarizer 转换为二元矩阵 ---
print("\n正在使用 MultiLabelBinarizer 转换标签为二元矩阵...")
mlb = MultiLabelBinarizer()
y_multilabel = mlb.fit_transform(parsed_labels)

print(f"标签二元矩阵 (Y) 生成完毕。")
print(f"Y 的形状: {y_multilabel.shape}") # (文档数, 唯一标签数)
print(f"共有 {len(mlb.classes_)} 个唯一的研究领域类别。")
print(f"类别名称 (前10个): {mlb.classes_[:10].tolist()}")

# --- 4. (可选) 分析标签分布 ---
print("\n研究领域标签分布统计 (出现次数最多的前15个):")
all_labels_flat = [label for sublist in parsed_labels for label in sublist]
label_counts = Counter(all_labels_flat)
for label, count in label_counts.most_common(15):
    print(f"  - {label}: {count} 次")

# --- 5. 保存处理好的标签矩阵和类别名称 ---
print(f"\n正在将标签二元矩阵保存到 '{output_labels_path}'...")
try:
    np.save(output_labels_path, y_multilabel)
    print(f"标签矩阵已成功保存。")
except Exception as e:
    print(f"保存标签矩阵时发生错误: {e}")
    exit()

print(f"\n正在将类别名称列表保存到 '{output_label_classes_path}'...")
try:
    with open(output_label_classes_path, 'wb') as f:
        pickle.dump(mlb.classes_, f)
    print(f"类别名称列表已成功保存。")
except Exception as e:
    print(f"保存类别名称列表时发生错误: {e}")
    exit()

# --- 验证 (可选) ---
print("\n尝试加载已保存的标签文件进行验证...")
try:
    loaded_y = np.load(output_labels_path)
    with open(output_label_classes_path, 'rb') as f:
        loaded_classes = pickle.load(f)
    print(f"成功加载标签矩阵，形状: {loaded_y.shape}")
    print(f"成功加载类别名称，数量: {len(loaded_classes)}")
    if loaded_y.size > 0 and len(loaded_classes) > 0:
        print(f"第一个文档的标签向量 (前10个维度): {loaded_y[0, :min(10, loaded_y.shape[1])]}")
        print(f"加载后标签矩阵的数据类型: {loaded_y.dtype}")
        print(f"加载的类别名称 (前5个): {list(loaded_classes[:5])}")

except Exception as e:
    print(f"加载验证文件时发生错误: {e}")

print("\n标签数据准备完成。")