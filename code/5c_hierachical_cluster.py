import numpy as np
import os
import pickle
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# --- 中文字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 配置与路径定义 ---
embedding_type = 'bert' 
linkage_method = 'ward' # 常用的链接方法

base_data_path = r'D:\Codes\textmining\data'
embeddings_input_dir = os.path.join(base_data_path, 'embeddings')
embedding_file_path = os.path.join(embeddings_input_dir, f'{embedding_type}_embeddings.npy')

output_results_dir = os.path.join(base_data_path, '..', 'results', 'clustering')
dendrogram_plot_path = os.path.join(output_results_dir, f'hierarchical_dendrogram_{embedding_type}_{linkage_method}.png')

# --- 创建输出目录 (如果不存在) ---
if not os.path.exists(output_results_dir):
    os.makedirs(output_results_dir)
    print(f"目录已创建: {output_results_dir}")

# --- 1. 加载BERT嵌入数据 ---
print(f"\n--- 正在加载 '{embedding_type}' 嵌入数据 ---")
if not os.path.exists(embedding_file_path):
    print(f"错误: 嵌入文件 '{embedding_file_path}' 未找到。")
    exit()
try:
    X_features = np.load(embedding_file_path)
    print(f"'{embedding_type}' 嵌入数据加载成功，形状: {X_features.shape}")
    if X_features.ndim == 1:
        X_features = X_features.reshape(-1, 1)
    num_documents = X_features.shape[0]
    if num_documents < 2 :
         print(f"错误: 样本数量 ({num_documents}) 不足以进行聚类。")
         exit()
except Exception as e:
    print(f"加载嵌入数据时发生错误: {e}")
    exit()

# --- 2. 执行层次聚类 ---
# 对于大型数据集，直接在完整数据上进行层次聚类可能非常耗时耗内存。
# 973个样本，BERT嵌入768维，ward方法计算量较大。如果太慢，可以考虑先降维或对样本抽样。
# 但我们先尝试在完整数据上运行。
print(f"\n--- 正在执行层次聚类 (方法: {linkage_method}) ---")
print("这可能需要一些时间，请耐心等待...")
try:
    # linkage_matrix的每一行代表一次合并：[idx1, idx2, distance, num_items_in_new_cluster]
    linkage_matrix = linkage(X_features, method=linkage_method, metric='euclidean')
    print("层次聚类链接矩阵计算完成。")
except Exception as e:
    print(f"执行层次聚类时发生错误: {e}")
    exit()

# --- 3. 绘制并保存树状图 ---
print("\n--- 正在绘制树状图 ---")
plt.figure(figsize=(20, 10)) # 可能需要较大的图才能看清
plt.title(f'层次聚类树状图 (BERT嵌入, {linkage_method}链接法)', fontsize=16)
plt.xlabel('文献样本索引 (或簇索引)')
plt.ylabel('距离 (Ward方差)')

# 由于样本量较大(973)，完整显示所有叶节点会非常拥挤。
# 使用 truncate_mode='lastp' 来显示最后形成的p个簇。
# 或者使用 truncate_mode='level' 来显示指定层级。
# 我们先尝试显示最后30个合并的簇。
num_last_clusters_to_show = 30 
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # 只显示最后p个非单元素形成的簇
    p=num_last_clusters_to_show,      # 显示最后形成的p个簇
    leaf_rotation=90.,      # 叶标签旋转90度
    leaf_font_size=8.,      # 叶标签字体大小
    show_contracted=True,   # 显示被截断的子树的大小
)

# 如果想要看到更多细节，可以调整p的值，或者不进行截断（但可能非常密集）
# 不截断的示例 (如果样本少可以尝试):
# dendrogram(linkage_matrix, leaf_rotation=90., leaf_font_size=8.)

plt.axhline(y=100, color='r', linestyle='--') # 示例：画一条横线用于可能的切割点参考 (y值需要根据实际距离调整)
# 您可以根据实际生成的树状图的距离范围来调整这条参考线的位置，或者不画

plt.tight_layout()
try:
    plt.savefig(dendrogram_plot_path)
    print(f"树状图已保存到: {dendrogram_plot_path}")
except Exception as e:
    print(f"保存树状图时发生错误: {e}")
plt.show()

print(f"\n--- 层次聚类 (使用 '{embedding_type}' 嵌入) 的处理已完成 ---")
print(f"请查看生成的树状图 '{dendrogram_plot_path}'。")
print("您可以根据树状图的结构来判断合适数量的簇，或者在某个距离阈值上'切割'树以获得扁平化的簇分配。")