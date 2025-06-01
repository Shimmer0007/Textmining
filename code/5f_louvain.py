import numpy as np
import os
import pickle
import gensim
from gensim.corpora.dictionary import Dictionary
import networkx as nx
import community as community_louvain # Louvain community detection
from collections import defaultdict, Counter
from itertools import combinations # To get pairs of words

# --- 中文字体设置 (如果后续需要绘图) ---
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei'] 
# plt.rcParams['axes.unicode_minus'] = False 

# --- 配置与路径定义 ---
base_data_path = r'D:\Codes\textmining\data'
cleaned_text_path = os.path.join(base_data_path, 'processed', 'cleaned_text.pkl')
# 我们需要LDA训练时用的词典来确定节点，但这里先加载cleaned_text，再重新生成一个词典并过滤
# 或者，如果有一个已保存的、用于LDA的gensim词典文件，也可以加载那个。
# 为简单起见，我们基于cleaned_text重新构建并过滤，确保与LDA步骤的词汇处理类似。

output_results_dir = os.path.join(base_data_path, '..', 'results', 'community_detection')
community_results_path = os.path.join(output_results_dir, 'louvain_communities.txt')
graph_path = os.path.join(output_results_dir, 'word_cooccurrence_graph.gexf') # 保存图文件

# --- 创建输出目录 (如果不存在) ---
if not os.path.exists(output_results_dir):
    os.makedirs(output_results_dir)
    print(f"目录已创建: {output_results_dir}")

def main():
    # --- 1. 加载预处理后的文本数据 ---
    print(f"\n--- 正在加载清洗后的文本数据 从 '{cleaned_text_path}' ---")
    if not os.path.exists(cleaned_text_path):
        print(f"错误: 文件 '{cleaned_text_path}' 未找到。")
        return
    try:
        with open(cleaned_text_path, 'rb') as f:
            documents = pickle.load(f)
        print(f"成功加载 {len(documents)} 篇文档。")
        documents = [doc for doc in documents if doc]
        if not documents:
            print("错误：所有文档都为空，无法构建网络。")
            return
        print(f"用于构建网络的非空文档数量: {len(documents)}")
    except Exception as e:
        print(f"加载文本数据时发生错误: {e}")
        return

    # --- 2. 创建词典并过滤 (与LDA步骤类似，以获得有意义的词作为节点) ---
    print("\n--- 正在创建和过滤词典以确定网络节点 ---")
    try:
        dictionary = Dictionary(documents)
        # 过滤词典：出现文档数过少（<5）或占比过高（>50%）的词
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        dictionary.compactify()
        # 获取过滤后的词汇表
        vocabulary = list(dictionary.token2id.keys())
        if not vocabulary:
            print("错误：过滤后词汇表为空，无法构建网络节点。")
            return
        print(f"词典过滤完成，将使用 {len(vocabulary)} 个唯一词元作为网络节点。")
    except Exception as e:
        print(f"创建词典时发生错误: {e}")
        return

    # --- 3. 构建词共现网络 ---
    print("\n--- 正在构建词共现网络 ---")
    # 使用defaultdict来存储共现次数
    cooccurrence_counts = defaultdict(int)
    
    # 遍历每个文档，统计在词汇表中的词的共现
    for doc in documents:
        #只考虑在我们过滤后词汇表中的词
        doc_vocab_words = [word for word in doc if word in vocabulary]
        # 获取文档中词对的组合 (无序，不重复)
        for word1, word2 in combinations(sorted(list(set(doc_vocab_words))), 2): # sorted确保(a,b)和(b,a)一致
            cooccurrence_counts[(word1, word2)] += 1
            
    if not cooccurrence_counts:
        print("错误：未能构建任何词共现关系。检查文本数据和词汇表。")
        return

    # 创建NetworkX图
    G = nx.Graph()
    for (word1, word2), weight in cooccurrence_counts.items():
        if weight > 0: # 可以设置一个最小权重阈值，例如 > 1，来进一步过滤边
            G.add_edge(word1, word2, weight=weight)
            
    print(f"词共现网络构建完成：包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")
    
    # (可选) 移除孤立节点，Louvain对孤立节点不敏感，但可以使图更紧凑
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"移除孤立节点后：包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边。")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("错误：图为空，无法进行社区发现。")
        return

    # (可选) 保存图文件，例如GEXF格式，可以用Gephi等软件打开查看
    try:
        nx.write_gexf(G, graph_path)
        print(f"网络图已保存到: {graph_path}")
    except Exception as e:
        print(f"保存网络图时发生错误: {e}")


    # --- 4. 应用Louvain算法进行社区发现 ---
    print("\n--- 正在应用Louvain算法进行社区发现 ---")
    # Louvain算法返回一个字典，键是节点，值是节点所属的社区ID
    # 使用 best_partition 以获得最佳模块度的划分
    # random_state用于确保结果可复现（如果算法内部有随机性）
    try:
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
        num_communities = len(set(partition.values()))
        print(f"Louvain算法完成，发现了 {num_communities} 个社区。")
        
        # 计算模块度
        modularity = community_louvain.modularity(partition, G, weight='weight')
        print(f"网络的模块度为: {modularity:.4f}")

    except Exception as e:
        print(f"运行Louvain算法时发生错误: {e}")
        return

    # --- 5. 展示与保存社区结果 ---
    print("\n--- 社区发现结果 (每个社区展示部分词语) ---")
    # 将社区结果整理为 社区ID -> [词列表] 的形式
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)

    report_lines = [f"Louvain Community Detection Results\nFound {num_communities} communities with modularity: {modularity:.4f}\n"]
    
    # 为了更好地展示，可以按社区大小排序或展示每个社区的核心词（例如，基于词在社区内的度数）
    # 这里简单地打印每个社区的部分成员
    sorted_communities = sorted(communities.items(), key=lambda item: len(item[1]), reverse=True) # 按社区大小排序

    for comm_id, nodes in sorted_communities:
        community_size = len(nodes)
        # 为了可读性，只显示社区的前N个词，或者最重要的词
        # 获取社区内部的子图，计算词的度数等来确定重要性，这里简化处理
        display_nodes = nodes[:20] # 每个社区最多显示20个词
        nodes_str = ", ".join(display_nodes)
        if len(nodes) > 20:
            nodes_str += "..."
        
        line = f"\n社区 #{comm_id} (大小: {community_size} 个词):\n  {nodes_str}"
        print(line)
        report_lines.append(line)

    try:
        with open(community_results_path, 'w', encoding='utf-8') as f:
            for line in report_lines:
                f.write(line + "\n")
        print(f"\n社区发现结果已保存到: {community_results_path}")
    except Exception as e:
        print(f"保存社区发现结果时发生错误: {e}")
        
    print("\n--- 社区发现处理完成 ---")

if __name__ == '__main__':
    main()