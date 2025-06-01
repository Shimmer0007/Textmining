import numpy as np
import os
import pickle
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
import logging
from multiprocessing import freeze_support # Added for safety, though if __name__ is main fix

# 配置日志，gensim会输出一些信息
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
gensim_logger = logging.getLogger("gensim")
gens_logger_level = getattr(logging, os.environ.get("GENSIM_LOGGING", "WARNING").upper(), None) # Get level from env var
if gens_logger_level is None: # If not set or invalid, default to WARNING
    gens_logger_level = logging.WARNING
gensim_logger.setLevel(gens_logger_level)


# --- 中文字体设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 配置与路径定义 ---
base_data_path = r'D:\Codes\textmining\data'
cleaned_text_path = os.path.join(base_data_path, 'processed', 'cleaned_text.pkl')

output_results_dir = os.path.join(base_data_path, '..', 'results', 'topic_modeling')
lda_tuning_plot_path = os.path.join(output_results_dir, 'lda_tuning_metrics.png')

def load_data(cleaned_text_path_func):
    print(f"\n--- 正在加载清洗后的文本数据 从 '{cleaned_text_path_func}' ---")
    if not os.path.exists(cleaned_text_path_func):
        print(f"错误: 文件 '{cleaned_text_path_func}' 未找到。")
        return None
    try:
        with open(cleaned_text_path_func, 'rb') as f:
            documents_func = pickle.load(f)
        print(f"成功加载 {len(documents_func)} 篇文档。")
        documents_func = [doc for doc in documents_func if doc] # 确保文档不为空列表
        if not documents_func:
            print("错误：所有文档都为空，无法进行LDA建模。")
            return None
        print(f"用于LDA的非空文档数量: {len(documents_func)}")
        return documents_func
    except Exception as e:
        print(f"加载文本数据时发生错误: {e}")
        return None

def prepare_lda_input(documents_func):
    print("\n--- 正在创建词典和语料库 ---")
    try:
        dictionary_func = Dictionary(documents_func)
        dictionary_func.filter_extremes(no_below=5, no_above=0.5)
        dictionary_func.compactify()
        print(f"词典创建完成，包含 {len(dictionary_func)} 个唯一词元。")

        corpus_func = [dictionary_func.doc2bow(doc) for doc in documents_func]
        corpus_func = [doc for doc in corpus_func if doc] # 过滤空BoW
        if not corpus_func:
            print("错误：所有文档在词典过滤后都变为空的BoW表示，无法进行LDA建模。")
            return None, None
        print(f"语料库创建完成，包含 {len(corpus_func)} 个BoW文档。")
        return dictionary_func, corpus_func
    except Exception as e:
        print(f"创建词典或语料库时发生错误: {e}")
        return None, None

def evaluate_lda_models(corpus_func, dictionary_func, documents_func, topics_range_func, passes_func, iterations_func, random_state_lda_func):
    print("\n--- 正在计算不同主题数量下的评估指标 ---")
    print(f"将测试的主题数量: {topics_range_func}")
    
    log_perplexity_values_func = []
    coherence_values_cv_func = []

    for num_topics_val in topics_range_func:
        print(f"  正在训练LDA模型 (主题数 = {num_topics_val})...")
        try:
            lda_model = LdaModel(corpus=corpus_func,
                                 id2word=dictionary_func,
                                 num_topics=num_topics_val,
                                 random_state=random_state_lda_func,
                                 passes=passes_func,
                                 iterations=iterations_func,
                                 chunksize=100,
                                 alpha='auto',
                                 eta='auto',
                                 eval_every=None) 
            
            current_log_perplexity = lda_model.log_perplexity(corpus_func) 
            log_perplexity_values_func.append(current_log_perplexity)
            
            coherence_model_cv = CoherenceModel(model=lda_model, texts=documents_func, dictionary=dictionary_func, coherence='c_v', processes=1) # Explicitly set processes=1
            coherence_cv = coherence_model_cv.get_coherence()
            coherence_values_cv_func.append(coherence_cv)
            
            print(f"    主题数 = {num_topics_val}, 对数困惑度(log_likelihood/word) = {current_log_perplexity:.4f}, 主题一致性 (C_v) = {coherence_cv:.4f}")
            
        except Exception as e:
            print(f"    训练LDA模型 (主题数 = {num_topics_val}) 时发生错误: {e}")
            log_perplexity_values_func.append(np.nan)
            coherence_values_cv_func.append(np.nan)
            continue
    return log_perplexity_values_func, coherence_values_cv_func

def plot_lda_metrics(topics_range_func, log_perplexity_values_func, coherence_values_cv_func, plot_path_func):
    print("\n--- 正在绘制评估指标图表 ---")
    fig, ax1 = plt.subplots(figsize=(12, 7))

    color = 'tab:red'
    ax1.set_xlabel('主题数量 (Number of Topics)')
    ax1.set_ylabel('平均对数似然 (Log Perplexity per word - 值越大越好)', color=color)
    ax1.plot(topics_range_func, log_perplexity_values_func, marker='o', color=color, label='平均对数似然')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(topics_range_func)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('主题一致性得分 (Coherence C_v - 值越大越好)', color=color) 
    ax2.plot(topics_range_func, coherence_values_cv_func, marker='x', linestyle='--', color=color, label='主题一致性 (C_v)')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('LDA模型主题数量选择：困惑度与一致性分析', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')

    try:
        plt.savefig(plot_path_func)
        print(f"LDA调优指标图已保存到 '{plot_path_func}'")
    except Exception as e:
        print(f"保存LDA调优图时发生错误: {e}")
    plt.show(block=False) # Use block=False for non-blocking show if run in some IDEs

def main():
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)
        print(f"目录已创建: {output_results_dir}")

    documents = load_data(cleaned_text_path)
    if documents is None:
        return

    dictionary, corpus = prepare_lda_input(documents)
    if dictionary is None or corpus is None:
        return

    min_topics = 2
    max_topics = 31 
    step_size = 3
    topics_range = list(range(min_topics, max_topics, step_size))
    if max_topics not in topics_range and (topics_range and max_topics > topics_range[-1]):
        potential_last_topic = max_topics - (max_topics % step_size) + (step_size if (max_topics % step_size) > (step_size/2.0) else 0)
        if potential_last_topic > topics_range[-1]:
             topics_range.append(potential_last_topic)
    topics_range = sorted(list(set(t for t in topics_range if t>=min_topics))) # Ensure unique, sorted, and >= min_topics

    passes = 15
    iterations = 100
    random_state_lda = 42

    log_perplexity_values, coherence_values_cv = evaluate_lda_models(
        corpus, dictionary, documents, topics_range, 
        passes, iterations, random_state_lda
    )

    plot_lda_metrics(topics_range, log_perplexity_values, coherence_values_cv, lda_tuning_plot_path)

    valid_coherence_scores = [(k, score) for k, score in zip(topics_range, coherence_values_cv) if not np.isnan(score)]
    if valid_coherence_scores:
        best_k_coherence = max(valid_coherence_scores, key=lambda item: item[1])
        print(f"\n根据主题一致性 (C_v) 分析，主题数 K = {best_k_coherence[0]} 时一致性得分最高，为 {best_k_coherence[1]:.4f}")
    else:
        print("\n未能计算出有效的主题一致性得分。")

    print("\nLDA主题数量选择分析完成。请查看图表和输出，选择一个合适的主题数量用于后续的最终LDA模型训练。")

if __name__ == '__main__':
    # freeze_support() # Typically needed if you're freezing the app (e.g. with PyInstaller)
    # For scripts, sometimes helpful on Windows if processes are started aggressively.
    # Try without it first, add if issues persist specifically tied to freezing or certain envs.
    # The primary fix is the if __name__ == '__main__' guard itself.
    # Also explicitly setting processes=1 in CoherenceModel if that's a source of issues.
    main()