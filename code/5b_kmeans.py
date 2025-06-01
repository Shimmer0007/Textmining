import numpy as np
import pandas as pd
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter

# --- 中文字体设置 (如果需要在脚本内生成图表，但此脚本主要输出文本和CSV) ---
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False 

# --- 配置与路径定义 ---
embedding_type = 'bert' 
chosen_k = 4 # 您选择的K值

base_data_path = r'D:\Codes\textmining\data'
embeddings_input_dir = os.path.join(base_data_path, 'embeddings')
embedding_file_path = os.path.join(embeddings_input_dir, f'{embedding_type}_embeddings.npy')

# 加载用于解读簇的标签信息
labels_input_path = os.path.join(base_data_path, 'processed', 'multilabel_targets_research_areas.npy')
labels_classes_path = os.path.join(base_data_path, 'processed', 'research_areas_classes.pkl')

output_results_dir = os.path.join(base_data_path, '..', 'results', 'clustering')
kmeans_clusters_csv_path = os.path.join(output_results_dir, f'kmeans_clusters_{embedding_type}_k{chosen_k}.csv')
kmeans_report_path = os.path.join(output_results_dir, f'kmeans_report_{embedding_type}_k{chosen_k}.txt')


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
except Exception as e:
    print(f"加载嵌入数据时发生错误: {e}")
    exit()

# --- 2. 执行K-Means聚类 ---
print(f"\n--- 正在使用 K={chosen_k} 执行K-Means聚类 ---")
kmeans = KMeans(n_clusters=chosen_k, 
                n_init='auto', 
                random_state=42,
                verbose=0)
cluster_labels = kmeans.fit_predict(X_features)
print("K-Means聚类完成。")

# --- 3. 计算并报告轮廓系数 ---
silhouette_avg = silhouette_score(X_features, cluster_labels)
print(f"\nK={chosen_k} 时的平均轮廓系数: {silhouette_avg:.4f}")

# --- 4. 保存簇分配结果到CSV ---
print(f"\n--- 正在保存簇分配结果到CSV文件 ---")
cluster_assignment_df = pd.DataFrame({
    'Document_ID': range(num_documents), # 文档ID为0到N-1的索引
    'Cluster_ID': cluster_labels
})
cluster_assignment_df.to_csv(kmeans_clusters_csv_path, index=False)
print(f"簇分配结果已保存到: {kmeans_clusters_csv_path}")

# --- 5. 簇的可解释性分析 (基于Research Areas) ---
print("\n--- 正在分析每个簇的 Research Areas 分布 (Top 5) ---")
report_content = []
report_content.append(f"K-Means Clustering Report (Embedding: {embedding_type}, K={chosen_k})\n")
report_content.append(f"Average Silhouette Score: {silhouette_avg:.4f}\n")

if os.path.exists(labels_input_path) and os.path.exists(labels_classes_path):
    try:
        y_multilabel = np.load(labels_input_path)
        with open(labels_classes_path, 'rb') as f:
            class_names = pickle.load(f)
        
        if y_multilabel.shape[0] != num_documents:
            print("警告: 标签数量与文档数量不匹配，无法进行簇解释分析。")
            report_content.append("警告: 标签数量与文档数量不匹配，无法进行簇解释分析。\n")
        else:
            for i in range(chosen_k):
                cluster_mask = (cluster_labels == i)
                num_docs_in_cluster = np.sum(cluster_mask)
                report_content.append(f"\n簇 {i} (包含 {num_docs_in_cluster} 个文档):")
                print(f"\n簇 {i} (包含 {num_docs_in_cluster} 个文档):")
                
                if num_docs_in_cluster == 0:
                    report_content.append("  此簇中没有文档。")
                    print("  此簇中没有文档。")
                    continue
                    
                # 获取当前簇中文档的标签向量
                cluster_y_labels = y_multilabel[cluster_mask]
                # 统计每个Research Area在该簇中出现的次数
                label_counts_in_cluster = cluster_y_labels.sum(axis=0)
                
                # 将计数与Research Area名称对应起来
                research_area_counts = []
                for label_idx, count in enumerate(label_counts_in_cluster):
                    if count > 0: # 只考虑在该簇中出现过的Research Area
                        research_area_counts.append((class_names[label_idx], count))
                
                # 按出现次数降序排序
                research_area_counts.sort(key=lambda x: x[1], reverse=True)
                
                if not research_area_counts:
                    report_content.append("  此簇中的文档没有明确的Research Areas标签。")
                    print("  此簇中的文档没有明确的Research Areas标签。")
                else:
                    report_content.append("  出现频率最高的 Research Areas (Top 5):")
                    print("  出现频率最高的 Research Areas (Top 5):")
                    for area_name, count in research_area_counts[:5]:
                        percentage_in_cluster = (count / num_docs_in_cluster) * 100
                        report_content.append(f"    - {area_name}: {count} 次 ({percentage_in_cluster:.1f}% of docs in cluster)")
                        print(f"    - {area_name}: {count} 次 ({percentage_in_cluster:.1f}% of docs in cluster)")
    except Exception as e:
        print(f"加载标签数据或分析簇时发生错误: {e}")
        report_content.append(f"加载标签数据或分析簇时发生错误: {e}\n")
else:
    print("未找到标签数据文件，跳过簇的Research Areas分布分析。")
    report_content.append("未找到标签数据文件，跳过簇的Research Areas分布分析。\n")

# --- 6. 将分析报告保存到文本文件 ---
try:
    with open(kmeans_report_path, 'w', encoding='utf-8') as f:
        for line in report_content:
            f.write(line + "\n")
    print(f"\nK-Means分析报告已保存到: {kmeans_report_path}")
except Exception as e:
    print(f"保存K-Means分析报告时发生错误: {e}")

print(f"\n--- K-Means (K={chosen_k}) 聚类配合 '{embedding_type}' 嵌入的处理已完成 ---")