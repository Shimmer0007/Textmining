import pandas as pd
import numpy as np
import os
import torch
from transformers import BertTokenizer, BertModel
import logging

# 配置日志级别，transformers库会输出一些信息
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING) # 可以设为INFO或WARNING，避免过多日志

# --- 定义文件路径 ---
base_data_path = r'D:\Codes\textmining\data'
raw_data_path = os.path.join(base_data_path, 'raw', 'data.csv') # 原始数据路径
embeddings_dir = os.path.join(base_data_path, 'embeddings')
output_path_bert = os.path.join(embeddings_dir, 'bert_embeddings.npy')

# --- 检查GPU可用性，并设置device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {device}")

# --- 1. 加载原始数据 ---
print(f"\n正在从 '{raw_data_path}' 加载原始摘要数据...")
try:
    df = pd.read_csv(raw_data_path)
    if 'Abstract' not in df.columns:
        print("错误: CSV文件中未找到 'Abstract' 列。")
        exit()
    # 将摘要转换为列表，并处理可能的NaN值
    abstracts = df['Abstract'].fillna('').astype(str).tolist()
    print(f"成功加载 {len(abstracts)} 个原始摘要。")
    if not abstracts:
        print("错误：未加载到任何摘要。")
        exit()
except FileNotFoundError:
    print(f"错误: 文件 '{raw_data_path}' 未找到。")
    exit()
except Exception as e:
    print(f"加载 '{raw_data_path}' 时发生错误: {e}")
    exit()

# --- 2. 初始化BERT模型和分词器 ---
model_name = 'bert-base-uncased'
print(f"\n正在加载BERT分词器: '{model_name}'...")
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    print(f"正在加载BERT模型: '{model_name}'...")
    model = BertModel.from_pretrained(model_name)
    model.to(device) # 将模型移动到选定的设备 (GPU或CPU)
    model.eval()     # 将模型设置为评估模式 (这对于不进行训练的推理很重要)
    print("BERT模型和分词器加载完成。")
except Exception as e:
    print(f"加载BERT模型或分词器时发生错误: {e}")
    print("请确保已安装 'transformers' 和 'torch' (或 'tensorflow') 库，并且网络连接正常以便下载模型。")
    exit()

# --- 3. 生成文档向量 ---
print("\n正在为每个摘要生成BERT向量...")
print(f"共有 {len(abstracts)} 个摘要需要处理。这个过程可能需要较长时间，请耐心等待。")

bert_embeddings_list = []
max_len = 512 # BERT的最大序列长度

# 为了节省内存，我们可以分批处理，但对于约1000个文档，逐个处理在CPU上虽然慢，但内存通常可控。
# 如果遇到内存问题，可以考虑进一步优化为批处理。
with torch.no_grad(): # 在此模式下不计算梯度，节省内存和计算
    for i, abstract_text in enumerate(abstracts):
        if (i + 1) % 50 == 0: # 每处理50个摘要打印一次进度
            print(f"  已处理 {i+1}/{len(abstracts)} 个摘要...")

        if not abstract_text.strip(): # 如果摘要为空或只有空格
            # 对于bert-base-uncased, 输出维度是768
            bert_embeddings_list.append(np.zeros(model.config.hidden_size, dtype=np.float32))
            continue

        # 分词并转换为PyTorch张量
        # padding='max_length' 会将所有序列填充到max_length
        # truncation=True 会将超过max_length的序列截断
        inputs = tokenizer(abstract_text, 
                           return_tensors='pt', 
                           max_length=max_len, 
                           padding='max_length', # 或者True，配合tokenizer的pad_token
                           truncation=True)
        
        inputs = {k: v.to(device) for k, v in inputs.items()} # 将输入数据移动到设备

        outputs = model(**inputs)
        
        # 我们使用[CLS]标记的输出来代表整个句子的嵌入
        # last_hidden_state 的形状是 (batch_size, sequence_length, hidden_size)
        # [CLS] 标记是第一个标记，所以我们取 [:, 0, :]
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        bert_embeddings_list.append(cls_embedding)

# 将嵌入列表转换为NumPy数组
bert_embeddings_np = np.array(bert_embeddings_list, dtype=np.float32)

print("BERT文档向量生成完毕。")
print(f"生成的文档向量矩阵形状: {bert_embeddings_np.shape}") # (文档数, 768)

# --- 4. 保存结果 ---
print(f"\n正在将BERT文档向量保存到 '{output_path_bert}'...")
try:
    np.save(output_path_bert, bert_embeddings_np)
    print(f"BERT文档向量已成功保存。")
except Exception as e:
    print(f"保存BERT文档向量时发生错误: {e}")
    exit()

# --- 验证 (可选) ---
print("\n尝试加载已保存的BERT文档向量文件进行验证...")
try:
    loaded_bert_vectors = np.load(output_path_bert)
    print(f"成功加载 '{output_path_bert}'。")
    print(f"加载后的矩阵形状: {loaded_bert_vectors.shape}")
    if loaded_bert_vectors.size > 0:
        print(f"第一个文档向量的前5个维度: {loaded_bert_vectors[0, :5]}")
        print(f"加载后矩阵的数据类型: {loaded_bert_vectors.dtype}")
except Exception as e:
    print(f"加载验证文件时发生错误: {e}")

print("\nBERT表示处理完成。")