import pickle
import numpy as np
import os

# --- 定义文件路径 ---
base_data_path = r'D:\Codes\textmining\data'
processed_dir = os.path.join(base_data_path, 'processed')
embeddings_dir = os.path.join(base_data_path, 'embeddings')

cleaned_text_path = os.path.join(processed_dir, 'cleaned_text.pkl')
vocab_path = os.path.join(processed_dir, 'tfidf_feature_names.pkl') # 使用TF-IDF的特征名作为词汇表
output_path_one_hot = os.path.join(embeddings_dir, 'one_hot.npy')

# --- 创建输出目录 (如果不存在) ---
if not os.path.exists(embeddings_dir):
    os.makedirs(embeddings_dir)
    print(f"目录已创建: {embeddings_dir}")
else:
    print(f"目录已存在: {embeddings_dir}")

# --- 1. 加载预处理数据 ---
print(f"\n正在加载清洗后的文本数据 از '{cleaned_text_path}'...")
try:
    with open(cleaned_text_path, 'rb') as f:
        cleaned_texts_tokens = pickle.load(f)
    print(f"成功加载 {len(cleaned_texts_tokens)} 个文档的清洗后词元。")
except FileNotFoundError:
    print(f"错误: 文件 '{cleaned_text_path}' 未找到。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载 '{cleaned_text_path}' 时发生错误: {e}")
    exit()

print(f"\n正在加载词汇表 (TF-IDF特征名) 从 '{vocab_path}'...")
try:
    with open(vocab_path, 'rb') as f:
        # tfidf_feature_names.pkl 可能保存的是NumPy数组或Python列表
        # 我们将其转换为列表，如果它还不是的话
        vocabulary_list = list(pickle.load(f)) 
    print(f"成功加载词汇表，包含 {len(vocabulary_list)} 个特征词。")
    if not vocabulary_list:
        print("错误：词汇表为空。无法进行One-Hot编码。")
        exit()
except FileNotFoundError:
    print(f"错误: 文件 '{vocab_path}' 未找到。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载 '{vocab_path}' 时发生错误: {e}")
    exit()

# --- 2. 生成Multi-Hot向量 ---
print("\n正在生成Multi-Hot编码向量...")

# 为了快速查找，将词汇表转换为set和带索引的字典
vocab_set = set(vocabulary_list)
vocab_index_map = {word: i for i, word in enumerate(vocabulary_list)}

num_documents = len(cleaned_texts_tokens)
num_features = len(vocabulary_list)

# 初始化一个全零的NumPy数组来存储multi-hot向量
# 数据类型设为整数，因为是0或1
multi_hot_vectors = np.zeros((num_documents, num_features), dtype=np.int8) 

for doc_idx, tokens in enumerate(cleaned_texts_tokens):
    if doc_idx % 100 == 0 and doc_idx > 0: # 每处理100个文档打印一次进度
        print(f"  已处理 {doc_idx}/{num_documents} 个文档...")
    
    # 对于当前文档中的每个词元，如果它在我们的词汇表中，
    # 则在multi_hot_vectors中对应位置标记为1
    for token in tokens:
        if token in vocab_index_map: # 使用预先构建的map进行高效查找
            feature_idx = vocab_index_map[token]
            multi_hot_vectors[doc_idx, feature_idx] = 1

print("Multi-Hot编码向量生成完毕。")
print(f"生成的向量矩阵形状: {multi_hot_vectors.shape}") # (文档数, 词汇表大小)

# --- 3. 保存结果 ---
print(f"\n正在将Multi-Hot向量保存到 '{output_path_one_hot}'...")
try:
    np.save(output_path_one_hot, multi_hot_vectors)
    print(f"Multi-Hot编码向量已成功保存。")
except Exception as e:
    print(f"保存Multi-Hot向量时发生错误: {e}")
    exit()

# --- 验证 (可选) ---
print("\n尝试加载已保存的One-Hot文件进行验证...")
try:
    loaded_one_hot_vectors = np.load(output_path_one_hot)
    print(f"成功加载 '{output_path_one_hot}'。")
    print(f"加载后的矩阵形状: {loaded_one_hot_vectors.shape}")
    # 可以打印一些统计信息，比如每篇文档的特征数量（向量中1的个数）
    if num_documents > 0 and loaded_one_hot_vectors.size > 0 :
        print(f"第一个文档的Multi-Hot向量中1的个数: {np.sum(loaded_one_hot_vectors[0])}")
        # 检查数据类型是否符合预期
        print(f"加载后矩阵的数据类型: {loaded_one_hot_vectors.dtype}")

except Exception as e:
    print(f"加载验证文件时发生错误: {e}")

print("\nOne-Hot编码处理完成。")