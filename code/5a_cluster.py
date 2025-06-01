import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# --- 中文字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 配置与路径定义 ---
embedding_type = 'bert' 
base_data_path = r'D:\Codes\textmining\data'
embeddings_input_dir = os.path.join(base_data_path, 'embeddings')
embedding_file_path = os.path.join(embeddings_input_dir, f'{embedding_type}_embeddings.npy')

output_results_dir = os.path.join(base_data_path, '..', 'results', 'clustering')
silhouette_plot_path = os.path.join(output_results_dir, f'kmeans_silhouette_analysis_{embedding_type}.png')

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
    if X_features.ndim == 1: # 如果是一维数组，尝试重塑
        X_features = X_features.reshape(-1, 1)
    if X_features.shape[0] < 2 : # 至少需要2个样本才能聚类
         print(f"错误: 样本数量 ({X_features.shape[0]}) 不足以进行聚类。")
         exit()
except Exception as e:
    print(f"加载嵌入数据时发生错误: {e}")
    exit()

# --- 2. 执行轮廓系数分析 ---
print("\n--- 正在执行K-Means轮廓系数分析 ---")
k_range = range(2, 21) # 测试K从2到20
silhouette_scores = []
inertia_values = [] # 同时记录inertia，可以辅助观察肘部

# 由于KMeans对于大型数据集或者高维数据可能较慢，特别是轮廓系数计算
# 如果数据量特别大，可以考虑对数据进行采样，但973个样本应该还好
print(f"将为 K 值范围 {list(k_range)} 计算轮廓系数和Inertia...")

for k_val in k_range:
    print(f"  正在测试 K = {k_val}...")
    kmeans = KMeans(n_clusters=k_val, 
                    n_init='auto', # 使用 'auto' 以避免警告，根据sklearn版本可能默认为10
                    random_state=42,
                    verbose=0) # 设置verbose=0减少不必要的输出
    cluster_labels = kmeans.fit_predict(X_features)
    
    # 轮廓系数至少需要2个簇，并且标签不能都一样
    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(X_features, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    else: # 如果所有点都分到一个簇（通常K=1时，但我们从K=2开始），或出现其他问题
        silhouette_scores.append(-1) # 赋一个无效值或标记
        print(f"    警告: K={k_val} 时，所有样本被分到了一个簇，或无法计算轮廓系数。")
        
    inertia_values.append(kmeans.inertia_)
    print(f"    K = {k_val}, 平均轮廓系数 = {silhouette_scores[-1]:.4f}, Inertia = {inertia_values[-1]:.2f}")

# --- 3. 绘制轮廓系数和肘部法则图 ---
print("\n--- 正在绘制分析图表 ---")
fig, ax1 = plt.subplots(figsize=(12, 7))

# 绘制轮廓系数
color = 'tab:red'
ax1.set_xlabel('簇的数量 (K)')
ax1.set_ylabel('平均轮廓系数', color=color)
ax1.plot(list(k_range), silhouette_scores, marker='o', color=color, label='平均轮廓系数')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(list(k_range))
ax1.grid(True, linestyle='--', alpha=0.7)

# 创建第二个Y轴共享X轴，用于绘制Inertia (肘部法则)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('簇内平方和 (Inertia)', color=color) 
ax2.plot(list(k_range), inertia_values, marker='x', linestyle='--', color=color, label='Inertia (肘部法则)')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('K-Means聚类性能分析：轮廓系数与肘部法则\n(使用BERT嵌入)', fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以适应标题

# 添加图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center right')

plt.savefig(silhouette_plot_path)
print(f"轮廓系数与肘部法则分析图已保存到 '{silhouette_plot_path}'")
plt.show()

# 打印找到的最佳轮廓系数对应的K值
if silhouette_scores:
    # 过滤掉之前可能设置的无效值-1
    valid_scores = [(k, score) for k, score in zip(k_range, silhouette_scores) if score > -1]
    if valid_scores:
        best_k_silhouette = max(valid_scores, key=lambda item: item[1])
        print(f"\n根据轮廓系数分析，K = {best_k_silhouette[0]} 时轮廓系数最高，为 {best_k_silhouette[1]:.4f}")
    else:
        print("\n未能计算出有效的轮廓系数。")
else:
    print("\n未能计算轮廓系数。")

print("\n轮廓系数分析完成。请查看图表和输出，选择一个合适的K值用于后续的K-Means聚类。")