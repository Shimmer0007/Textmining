import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- 定义基础路径 ---
base_data_path = r'D:\Codes\textmining\data'
embeddings_input_dir = os.path.join(base_data_path, 'embeddings')
labels_input_path = os.path.join(base_data_path, 'processed', 'multilabel_targets_research_areas.npy')
classification_data_output_dir = os.path.join(base_data_path, 'classification_data')

# --- 检查标签文件是否存在 ---
if not os.path.exists(labels_input_path):
    print(f"错误: 标签文件 '{labels_input_path}' 未找到。请先运行标签准备脚本。")
    exit()

# --- 加载通用的标签数据 (Y) ---
print(f"正在加载标签数据 (Y) 从 '{labels_input_path}'...")
try:
    y_labels = np.load(labels_input_path)
    print(f"标签数据加载成功，形状: {y_labels.shape}")
except Exception as e:
    print(f"加载标签数据时发生错误: {e}")
    exit()

# --- 定义要处理的嵌入类型及其文件名 ---
embedding_files_info = {
    'one_hot': 'one_hot.npy',
    'word2vec_skipgram': 'word2vec_skipgram_embeddings.npy',
    'word2vec_cbow': 'word2vec_cbow_embeddings.npy',
    'bert': 'bert_embeddings.npy'
}

# --- 划分参数 ---
test_set_size = 0.2  # 20% 的数据作为测试集
random_seed = 42     # 随机种子，确保划分可复现

# --- 遍历每种嵌入类型，进行划分并保存 ---
for embedding_name, embedding_filename in embedding_files_info.items():
    print(f"\n--- 正在处理嵌入类型: {embedding_name} ---")
    
    embedding_file_path = os.path.join(embeddings_input_dir, embedding_filename)
    
    # 检查嵌入文件是否存在
    if not os.path.exists(embedding_file_path):
        print(f"  警告: 嵌入文件 '{embedding_file_path}' 未找到。跳过此嵌入类型。")
        continue
        
    # 加载特定的嵌入数据 (X)
    print(f"  正在加载嵌入数据 (X) 从 '{embedding_file_path}'...")
    try:
        x_features = np.load(embedding_file_path)
        print(f"  '{embedding_name}' 嵌入数据加载成功，形状: {x_features.shape}")
        
        # 校验X和Y的样本数量是否一致
        if x_features.shape[0] != y_labels.shape[0]:
            print(f"  错误: 特征数量 ({x_features.shape[0]}) 与标签数量 ({y_labels.shape[0]}) 不匹配。")
            print(f"  请检查 '{embedding_filename}' 和标签文件。跳过此嵌入类型。")
            continue
            
    except Exception as e:
        print(f"  加载 '{embedding_filename}' 时发生错误: {e}。跳过此嵌入类型。")
        continue
        
    # 创建该嵌入类型的输出子目录
    current_output_subdir = os.path.join(classification_data_output_dir, embedding_name)
    if not os.path.exists(current_output_subdir):
        os.makedirs(current_output_subdir)
        print(f"  目录已创建: {current_output_subdir}")
    else:
        print(f"  目录已存在: {current_output_subdir}")
        
    # 执行训练集和测试集的划分
    # 对于多标签数据，标准的train_test_split不直接支持分层，但仍可用于随机划分
    print(f"  正在划分数据 (test_size={test_set_size}, random_state={random_seed})...")
    X_train, X_test, y_train, y_test = train_test_split(
        x_features, y_labels, 
        test_size=test_set_size, 
        random_state=random_seed
    )
    
    print(f"  划分完成:")
    print(f"    X_train 形状: {X_train.shape}, y_train 形状: {y_train.shape}")
    print(f"    X_test 形状: {X_test.shape}, y_test 形状: {y_test.shape}")
    
    # 保存划分后的数据
    print(f"  正在保存划分后的数据到 '{current_output_subdir}'...")
    try:
        np.save(os.path.join(current_output_subdir, 'X_train.npy'), X_train)
        np.save(os.path.join(current_output_subdir, 'X_test.npy'), X_test)
        np.save(os.path.join(current_output_subdir, 'y_train.npy'), y_train)
        np.save(os.path.join(current_output_subdir, 'y_test.npy'), y_test)
        print(f"  数据已成功保存。")
    except Exception as e:
        print(f"  保存数据时发生错误: {e}")

print("\n所有嵌入类型的数据划分处理完成。")