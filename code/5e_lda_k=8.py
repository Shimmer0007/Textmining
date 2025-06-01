import numpy as np
import os
import pickle
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import logging

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
gensim_logger = logging.getLogger("gensim")
gens_logger_level = getattr(logging, os.environ.get("GENSIM_LOGGING", "WARNING").upper(), None)
if gens_logger_level is None:
    gens_logger_level = logging.WARNING
gensim_logger.setLevel(gens_logger_level)

# --- 配置与路径定义 ---
num_final_topics = 8  # 您选择的主题数量
embedding_type_source = 'cleaned_text' # 标记输入源，非嵌入

base_data_path = r'D:\Codes\textmining\data'
cleaned_text_path = os.path.join(base_data_path, 'processed', 'cleaned_text.pkl')

output_models_dir = os.path.join(base_data_path, '..', 'models') 
output_results_dir = os.path.join(base_data_path, '..', 'results', 'topic_modeling')
lda_model_save_path = os.path.join(output_models_dir, 'lda_model.pkl') # 按照您的要求
lda_topics_output_path = os.path.join(output_results_dir, f'lda_topics_k{num_final_topics}.txt')
lda_final_metrics_path = os.path.join(output_results_dir, f'lda_final_metrics_k{num_final_topics}.txt')

# --- 创建输出目录 (如果不存在) ---
for dir_path in [output_models_dir, output_results_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录已创建: {dir_path}")

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
            print("错误：所有文档都为空，无法进行LDA建模。")
            return
        print(f"用于LDA的非空文档数量: {len(documents)}")
    except Exception as e:
        print(f"加载文本数据时发生错误: {e}")
        return

    # --- 2. 准备LDA的输入：词典和语料库 ---
    print("\n--- 正在创建词典和语料库 ---")
    try:
        dictionary = Dictionary(documents)
        dictionary.filter_extremes(no_below=5, no_above=0.5)
        dictionary.compactify()
        print(f"词典创建完成，包含 {len(dictionary)} 个唯一词元。")

        corpus = [dictionary.doc2bow(doc) for doc in documents]
        corpus = [doc for doc in corpus if doc]
        if not corpus:
            print("错误：所有文档在词典过滤后都变为空的BoW表示，无法进行LDA建模。")
            return
        print(f"语料库创建完成，包含 {len(corpus)} 个BoW文档。")
    except Exception as e:
        print(f"创建词典或语料库时发生错误: {e}")
        return

    # --- 3. 训练最终的LDA模型 ---
    print(f"\n--- 正在训练LDA模型 (主题数 = {num_final_topics}) ---")
    # LDA模型参数 (与调优时保持一致)
    passes = 15
    iterations = 100
    random_state_lda = 42 
    # 对于最终模型，可以考虑增加 passes 和 iterations 以获得更稳定的结果，但会增加时间
    # 例如: passes=20, iterations=400

    try:
        lda_final_model = LdaModel(corpus=corpus,
                                   id2word=dictionary,
                                   num_topics=num_final_topics,
                                   random_state=random_state_lda,
                                   passes=passes,
                                   iterations=iterations,
                                   chunksize=100,
                                   alpha='auto',
                                   eta='auto',
                                   eval_every=None) # 设为None, 手动评估
        print("最终LDA模型训练完成。")
    except Exception as e:
        print(f"训练最终LDA模型时发生错误: {e}")
        return

    # --- 4. 查看和保存主题 ---
    print(f"\n--- LDA模型发现的 {num_final_topics} 个主题 (每个主题前10个词) ---")
    topics_formatted = []
    # num_words参数控制每个主题显示多少个词
    # formatted=True 返回 (词*权重 + ...) 的字符串，formatted=False 返回 (词, 权重) 的元组列表
    shown_topics = lda_final_model.print_topics(num_topics=num_final_topics, num_words=10)
    for topic_id, topic_content in shown_topics:
        topic_line = f"主题 #{topic_id}: {topic_content}"
        print(topic_line)
        topics_formatted.append(topic_line)
    
    try:
        with open(lda_topics_output_path, 'w', encoding='utf-8') as f:
            f.write(f"LDA Model - {num_final_topics} Topics (Top 10 words per topic):\n")
            for line in topics_formatted:
                f.write(line + "\n")
        print(f"主题词分布已保存到: {lda_topics_output_path}")
    except Exception as e:
        print(f"保存主题词分布时发生错误: {e}")

    # --- 5. 评估最终模型 ---
    print("\n--- 正在评估最终LDA模型 ---")
    try:
        final_log_perplexity = lda_final_model.log_perplexity(corpus) # 值越大（越接近0）越好
        
        coherence_model_cv_final = CoherenceModel(model=lda_final_model, texts=documents, dictionary=dictionary, coherence='c_v', processes=1)
        final_coherence_cv = coherence_model_cv_final.get_coherence() # 值越大越好
        
        print(f"最终模型 (K={num_final_topics}):")
        print(f"  对数困惑度 (log_likelihood/word): {final_log_perplexity:.4f}")
        print(f"  主题一致性 (C_v): {final_coherence_cv:.4f}")

        # 保存评估指标
        with open(lda_final_metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"Final LDA Model Metrics (K={num_final_topics}):\n")
            f.write(f"Log Perplexity (log_likelihood/word): {final_log_perplexity:.4f}\n")
            f.write(f"Coherence Score (C_v): {final_coherence_cv:.4f}\n")
        print(f"最终评估指标已保存到: {lda_final_metrics_path}")

    except Exception as e:
        print(f"评估最终LDA模型时发生错误: {e}")


    # --- 6. 保存训练好的LDA模型 ---
    print("\n--- 正在保存训练好的LDA模型 ---")
    try:
        lda_final_model.save(lda_model_save_path)
        print(f"LDA模型已成功保存到: {lda_model_save_path}")
    except Exception as e:
        print(f"保存LDA模型时发生错误: {e}")

    print(f"\n--- LDA主题建模 (K={num_final_topics}) 处理完成 ---")

if __name__ == '__main__':
    main()