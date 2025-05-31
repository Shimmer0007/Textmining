import pickle
import numpy as np
import os
from gensim.models import Word2Vec
import logging

# 配置日志记录，gensim会输出训练过程信息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- 定义文件路径 ---
base_data_path = r'D:\Codes\textmining\data'
processed_dir = os.path.join(base_data_path, 'processed')
embeddings_dir = os.path.join(base_data_path, 'embeddings')

cleaned_text_path = os.path.join(processed_dir, 'cleaned_text.pkl')
output_path_word2vec = os.path.join(embeddings_dir, 'word2vec_embeddings.npy')
# (可选) 保存训练好的Word2Vec模型本身，以便后续使用或检查
model_output_path = os.path.join(embeddings_dir, 'word2vec.model')


# --- 1. 加载预处理数据 ---
print(f"\n正在加载清洗后的文本数据 از '{cleaned_text_path}'...")
try:
    with open(cleaned_text_path, 'rb') as f:
        # cleaned_texts_tokens 是一个列表的列表，例如: [['word1', 'word2'], ['word3', 'word4', 'word5']]
        cleaned_texts_tokens = pickle.load(f)
    print(f"成功加载 {len(cleaned_texts_tokens)} 个文档的清洗后词元。")
    # 过滤掉在预处理后可能变为空列表的文档，gensim期望的是非空列表的列表
    sentences_for_w2v = [tokens for tokens in cleaned_texts_tokens if tokens]
    if not sentences_for_w2v:
        print("错误：所有文档在清洗后都为空，无法训练Word2Vec模型。")
        exit()
    print(f"用于Word2Vec训练的非空文档数量: {len(sentences_for_w2v)}")

except FileNotFoundError:
    print(f"错误: 文件 '{cleaned_text_path}' 未找到。请确保文件路径正确。")
    exit()
except Exception as e:
    print(f"加载 '{cleaned_text_path}' 时发生错误: {e}")
    exit()

# --- 2. 训练Word2Vec模型 ---
print("\n正在训练Word2Vec模型...")
# Word2Vec模型参数
vector_dim = 100  # 词向量维度
window_size = 5   # 上下文窗口大小
min_word_count = 2 # 最小词频阈值
sg_choice = 1     # 1 for Skip-Gram, 0 for CBOW
num_workers = 4   # 并行处理的线程数 (根据CPU核心数调整)
training_epochs = 10 # 训练迭代次数

try:
    word2vec_model = Word2Vec(sentences=sentences_for_w2v,
                              vector_size=vector_dim,
                              window=window_size,
                              min_count=min_word_count,
                              sg=sg_choice,
                              workers=num_workers,
                              epochs=training_epochs) # gensim 4.x.x 使用 epochs
                              # 对于旧版gensim (3.x.x), 参数是 iter (不是epochs)
    
    print("Word2Vec模型训练完成。")
    print(f"词汇表大小: {len(word2vec_model.wv.key_to_index)}")

    # (可选) 保存训练好的Word2Vec模型
    word2vec_model.save(model_output_path)
    print(f"Word2Vec模型已保存到 '{model_output_path}'")

except Exception as e:
    print(f"训练Word2Vec模型时发生错误: {e}")
    # 可能是gensim版本问题，例如 epochs vs iter 参数
    print("提示: 如果是关于 'epochs' 或 'iter' 的错误，请检查您的gensim版本。")
    print("Gensim 4.x.x 使用 'epochs'，旧版本使用 'iter'。")
    exit()

# --- 3. 生成文档向量 (通过平均词向量) ---
print("\n正在为每个文档生成Word2Vec平均向量...")
document_vectors = []

for tokens in cleaned_texts_tokens: # 使用原始的 cleaned_texts_tokens 以保持与one-hot相同的文档数量
    word_vectors = []
    if not tokens: # 如果原始文档清洗后为空
        document_vectors.append(np.zeros(vector_dim, dtype=np.float32))
        continue

    for token in tokens:
        if token in word2vec_model.wv: # 检查词是否在模型的词汇表中
            word_vectors.append(word2vec_model.wv[token])
    
    if not word_vectors: # 如果文档中的所有词都不在模型词汇表中
        # 添加一个零向量
        document_vectors.append(np.zeros(vector_dim, dtype=np.float32))
    else:
        # 计算平均向量
        document_vectors.append(np.mean(word_vectors, axis=0))

# 将文档向量列表转换为NumPy数组
document_vectors_np = np.array(document_vectors)

print("文档向量生成完毕。")
print(f"生成的文档向量矩阵形状: {document_vectors_np.shape}") # (文档数, vector_dim)

# --- 4. 保存结果 ---
print(f"\n正在将Word2Vec文档向量保存到 '{output_path_word2vec}'...")
try:
    np.save(output_path_word2vec, document_vectors_np)
    print(f"Word2Vec文档向量已成功保存。")
except Exception as e:
    print(f"保存Word2Vec文档向量时发生错误: {e}")
    exit()

# --- 验证 (可选) ---
print("\n尝试加载已保存的Word2Vec文档向量文件进行验证...")
try:
    loaded_w2v_vectors = np.load(output_path_word2vec)
    print(f"成功加载 '{output_path_word2vec}'。")
    print(f"加载后的矩阵形状: {loaded_w2v_vectors.shape}")
    if loaded_w2v_vectors.size > 0:
         print(f"第一个文档向量的前5个维度: {loaded_w2v_vectors[0, :5]}")
         # 检查数据类型是否符合预期
         print(f"加载后矩阵的数据类型: {loaded_w2v_vectors.dtype}")
except Exception as e:
    print(f"加载验证文件时发生错误: {e}")

print("\nWord2Vec表示处理完成。")