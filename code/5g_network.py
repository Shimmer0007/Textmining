import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches # For creating legend patches
import numpy as np
import community as community_louvain  # python-louvain库
from collections import defaultdict

# --- 中文字符和负号显示 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 配置与路径 ---
graph_path = r'D:\Codes\textmining\results\community_detection\word_cooccurrence_graph.gexf'
output_image_path_original = r'D:\Codes\textmining\results\community_detection\network_visualization_original_legend.png'
output_image_path_aggregated = r'D:\Codes\textmining\results\community_detection\network_visualization_aggregated_legend.png'

# --- [ADJUSTABLE PARAMETERS for Aggregated Graph Visualization] ---
# --- 社区检测分辨率 ---
community_resolution = 1.2  # >1:更多小社区; <1:更少大社区; 1:默认

# --- 聚合图节点大小 ---
AGG_MIN_NODE_SIZE = 300   # 社区节点的最小尺寸
AGG_MAX_NODE_SIZE = 6000  # 社区节点的最大尺寸

# --- 聚合图边线粗细 ---
AGG_BASE_EDGE_WIDTH = 1.0    # 社区间连接线的最小粗细
AGG_MAX_SCALE_EDGE_FACTOR = 7.0 # 社区间连接线粗细的最大缩放因子 (最终粗细 = BASE + SCALE_FACTOR * normalized_weight)

# --- 1. 加载网络图 ---
print("正在加载网络图...")
G_original = nx.read_gexf(graph_path)
print(f"原始网络加载完成：{G_original.number_of_nodes()}个节点，{G_original.number_of_edges()}条边")

if G_original.number_of_nodes() == 0:
    print("错误：加载的图为空，无法进行后续操作。请检查GEXF文件。")
    exit()

# --- 2. 社区检测 (用于着色和聚合) ---
print(f"正在进行社区检测 (基于原始图, resolution={community_resolution})...")
try:
    partition = community_louvain.best_partition(G_original, weight='weight', resolution=community_resolution)
except TypeError:
    print("Warning: community_louvain.best_partition did not accept 'weight' or 'resolution' key. Trying fallbacks.")
    try:
        partition = community_louvain.best_partition(G_original, resolution=community_resolution)
    except TypeError:
         try:
            partition = community_louvain.best_partition(G_original, weight='weight')
         except TypeError:
            partition = community_louvain.best_partition(G_original)


communities = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append(node)

num_detected_communities = len(communities)
print(f"社区检测完成，发现 {num_detected_communities} 个社区。")

if num_detected_communities == 0:
    print("错误：未检测到任何社区。请检查图数据或社区检测参数。")
elif num_detected_communities < 2 and G_original.number_of_nodes() > 1 :
    print(f"警告：只检测到 {num_detected_communities} 个社区。聚合图可能意义不大。")
    print("如果希望看到更多社区，请尝试增大 'community_resolution' 参数。")


# --- 3. 创建社区聚合图 (Meta-Graph) ---
print("正在创建社区聚合图...")
G_community = nx.Graph()

community_node_sizes_orig_count = {} # Store original node count for each community
if num_detected_communities > 0:
    for comm_id, nodes_in_comm in communities.items():
        G_community.add_node(comm_id, size_orig=len(nodes_in_comm))
        community_node_sizes_orig_count[comm_id] = len(nodes_in_comm)

    community_edge_weights = defaultdict(float)
    for u, v, data in G_original.edges(data=True):
        comm_u = partition.get(u)
        comm_v = partition.get(v)
        
        if comm_u is None or comm_v is None:
            continue
        weight = data.get('weight', 1.0)
        if comm_u != comm_v:
            edge = tuple(sorted((comm_u, comm_v)))
            community_edge_weights[edge] += weight

    for (c1, c2), agg_weight in community_edge_weights.items():
        G_community.add_edge(c1, c2, weight=agg_weight)

print(f"社区聚合图创建完成：{G_community.number_of_nodes()}个节点 (社区)，{G_community.number_of_edges()}条边 (社区间连接)")

# --- 4. 可视化聚合后的社区网络 ---
if G_community.number_of_nodes() > 0:
    print("正在绘制聚合后的社区网络...")
    # Create figure and axes for better control over legend placement
    fig, ax = plt.subplots(figsize=(22, 18), dpi=300) # Slightly wider for legend
    ax.axis('off')

    # Color mapping
    unique_community_ids = sorted(list(G_community.nodes()))
    colors_available = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values()) # Expanded color list
    community_color_map = {comm_id: colors_available[i % len(colors_available)] for i, comm_id in enumerate(unique_community_ids)}
    node_colors_agg = [community_color_map[n] for n in G_community.nodes()]

    # Node sizes (using parameters from top)
    if community_node_sizes_orig_count:
        max_orig_comm_size = max(community_node_sizes_orig_count.values()) if community_node_sizes_orig_count else 1
        if max_orig_comm_size == 0 : max_orig_comm_size = 1
        node_sizes_agg = [
            AGG_MIN_NODE_SIZE + (AGG_MAX_NODE_SIZE - AGG_MIN_NODE_SIZE) * (community_node_sizes_orig_count.get(n, 0) / max_orig_comm_size)
            for n in G_community.nodes()
        ]
    else:
        node_sizes_agg = [AGG_MIN_NODE_SIZE] * G_community.number_of_nodes() # Default size if no counts

    # Edge widths and alphas (using parameters from top)
    if G_community.number_of_edges() > 0:
        agg_weights = [d['weight'] for u, v, d in G_community.edges(data=True)]
        min_agg_w = min(agg_weights) if agg_weights else 1.0
        max_agg_w = max(agg_weights) if agg_weights else 1.0
        
        if max_agg_w == min_agg_w: # Handle case where all weights are the same
            edge_widths_agg = [AGG_BASE_EDGE_WIDTH + AGG_MAX_SCALE_EDGE_FACTOR / 2.0] * len(agg_weights)
        else:
            edge_widths_agg = [
                AGG_BASE_EDGE_WIDTH + AGG_MAX_SCALE_EDGE_FACTOR * ((w - min_agg_w) / (max_agg_w - min_agg_w))
                for w in agg_weights
            ]
        edge_alphas_agg = [0.3 + 0.7 * ((w - min_agg_w) / (max_agg_w - min_agg_w) if max_agg_w != min_agg_w else 0.5) for w in agg_weights]
    else:
        edge_widths_agg = []
        edge_alphas_agg = []

    # Layout
    k_val = 0.9 / np.sqrt(G_community.number_of_nodes()) if G_community.number_of_nodes() > 0 else 0.5
    pos_agg = nx.spring_layout(G_community, k=k_val, iterations=100, seed=42, weight='weight')

    # Draw nodes and edges
    nx.draw_networkx_nodes(
        G_community, pos_agg, ax=ax,
        node_size=node_sizes_agg,
        node_color=node_colors_agg,
        alpha=0.9
    )
    if G_community.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G_community, pos_agg, ax=ax,
            width=edge_widths_agg,
            alpha=edge_alphas_agg,
            edge_color='silver' # Was 'gray', 'silver' might look lighter
        )

    # --- Create Legend ---
    legend_handles = []
    for comm_id in unique_community_ids: # Iterate in sorted order for consistency
        label_text = f"社区 {comm_id} ({community_node_sizes_orig_count.get(comm_id, 0)}词)"
        color = community_color_map[comm_id]
        legend_handles.append(mpatches.Patch(color=color, label=label_text))
    
    # Place legend outside the plot on the right
    ax.legend(handles=legend_handles, title="社区图例 (大小与词数相关)",
              title_fontsize='14', fontsize='10',
              bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)


    fig.suptitle(f"词共现网络聚合视图 (社区为节点, Resolution={community_resolution})", fontsize=26)
    # Adjust layout to make space for legend and suptitle
    fig.tight_layout(rect=[0, 0, 0.85, 0.96]) # rect=[left, bottom, right, top]

    plt.savefig(output_image_path_aggregated, bbox_inches='tight', dpi=300) # bbox_inches='tight' can sometimes conflict with external legend.
                                                                            # If legend is clipped, remove bbox_inches='tight' or adjust rect in tight_layout.
    print(f"聚合网络可视化已保存至: {output_image_path_aggregated}")
    plt.close(fig) # Close the figure
else:
    print("聚合图为空或仅含单个节点，跳过聚合图的绘制。")


# --- [OPTIONAL] 5. 绘制原始网络 ---
draw_original_anyway = False
if G_original.number_of_nodes() < 1000 and (draw_original_anyway or num_detected_communities < 5) :
    print("正在绘制原始网络 (可能会很密集)...")
    fig_orig, ax_orig = plt.subplots(figsize=(24, 18), dpi=300)
    ax_orig.axis('off')

    if 'colors_available' not in locals(): # Ensure colors_available is defined
        colors_available = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    
    original_node_colors = [colors_available[partition.get(n, -1) % len(colors_available)] for n in G_original.nodes()]

    degrees_original = dict(G_original.degree())
    if degrees_original:
        max_degree_original = max(degrees_original.values()) if degrees_original else 1
        if max_degree_original == 0: max_degree_original = 1
        min_size_orig, max_size_orig = 20, 200 # Slightly larger original nodes too
        node_sizes_original = [min_size_orig + (max_size_orig - min_size_orig) * (degrees_original.get(n, 0) / max_degree_original) for n in G_original.nodes()]
    else:
        node_sizes_original = [30] * G_original.number_of_nodes()

    if G_original.number_of_edges() > 0:
        original_weights = [d.get('weight', 1.0) for u, v, d in G_original.edges(data=True)]
        min_orig_w = min(original_weights) if original_weights else 1.0
        max_orig_w = max(original_weights) if original_weights else 1.0
        if max_orig_w == 0: max_orig_w = 1.0

        edge_alphas_original = [0.05 + 0.6 * ((w - min_orig_w) / (max_orig_w - min_orig_w) if max_orig_w != min_orig_w else 0.5) for w in original_weights]
        edge_widths_original = [0.2 + 1.5 * ((w - min_orig_w) / (max_orig_w - min_orig_w) if max_orig_w != min_orig_w else 0.5) for w in original_weights] # Also slightly thicker edges for original

    else:
        edge_alphas_original = []
        edge_widths_original = []


    print("正在计算原始图布局 (可能耗时较长)...")
    k_orig_val = 0.3 / np.sqrt(G_original.number_of_nodes()) if G_original.number_of_nodes() > 0 else 0.1
    pos_original = nx.spring_layout(G_original, k=k_orig_val, iterations=30, seed=42, weight='weight')

    nx.draw_networkx_nodes(
        G_original, pos_original, ax=ax_orig,
        node_size=node_sizes_original,
        node_color=original_node_colors,
        alpha=0.7
    )
    if G_original.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G_original, pos_original, ax=ax_orig,
            width=edge_widths_original,
            alpha=edge_alphas_original,
            edge_color='lightgray'
        )

    if degrees_original:
        top_nodes_original = sorted(degrees_original.items(), key=lambda x: x[1], reverse=True)[:30] # Reduced labels for clarity
        nx.draw_networkx_labels(
            G_original, pos_original, ax=ax_orig,
            labels={n[0]: n[0] for n in top_nodes_original},
            font_size=8, # Slightly larger font for labels
            alpha=0.75
        )
    
    fig_orig.suptitle("原始词共现网络可视化 (按社区着色)", fontsize=24)
    fig_orig.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(output_image_path_original, bbox_inches='tight', dpi=300)
    print(f"原始网络可视化已保存至: {output_image_path_original}")
    plt.close(fig_orig)
else:
    print("原始图节点过多或聚合图已提供足够社区，跳过原始图的详细绘制以节约时间/资源。")

print("处理完成。")