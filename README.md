# 文本挖掘与文献分析：以“银发经济”领域为例

## 1. 项目概述

本项目是课程《统计机器学习与文本挖掘》的上机实验材料，通过对“银发经济”领域相关学术文献的摘要进行文本挖掘，探索该领域的研究主题、演化趋势以及不同文本分析技术的应用效果。实验内容涵盖了数据预处理、多种文本表示方法、文本分类、主题分析与社区发现等核心环节。

## 2. 数据来源与描述

* **数据来源**: Web of Science (WoS)核心合集数据库。
* **原始数据**: `data/raw/data.csv` 文件，包含了从WoS导出的文献信息，主要使用了文献摘要 (`Abstract`)、发表年份 (`Publication Year`) 和学科分类 (`Research Areas`, `WoS Categories`) 等字段。
* **数据规模**: 共973篇文献记录。

## 3. 实验流程与核心技术

主要包括以下几个阶段：

### 3.1 数据预处理 (对应报告第4节)
* 目标：清洗文本、提取特征词。
* 核心步骤：
    * 分词 (NLTK)。
    * 词形还原 (NLTK WordNetLemmatizer)。
    * 停用词处理 (移除通用停用词和低频词)。
    * TF-IDF筛选 (保留前800个关键词)。
* 输出文件：
    * `data/processed/cleaned_text.pkl` (清洗后词元列表)
    * `data/processed/tfidf_features.pkl` (TF-IDF特征矩阵)
    * `data/processed/tfidf_feature_names.pkl` (TF-IDF特征词)

### 3.2 文本表示 (对应报告第5节)
对比了多种文本表示方法，将预处理后的文本转换为数值向量：
* **One-Hot编码 (Multi-Hot)**: 基于TF-IDF筛选的800个特征词构建。
    * 输出: `data/embeddings/one_hot.npy` (973, 800)
* **Word2Vec**: 在清洗后的摘要上训练词向量模型 (100维)。
    * Skip-gram: `data/embeddings/word2vec_skipgram_embeddings.npy` (973, 100) 和 `word2vec_skipgram.model`
    * CBOW: `data/embeddings/word2vec_cbow_embeddings.npy` (973, 100) 和 `word2vec_cbow.model`
* **BERT**: 使用预训练的 `bert-base-uncased` 模型提取上下文相关的文档嵌入 (768维)。
    * 输出: `data/embeddings/bert_embeddings.npy` (973, 768)

### 3.3 文本分类 (对应报告第6节)
* **目标**: 基于文献的“Research Areas”进行多标签文本分类 (97个类别)。
* **标签处理**: 使用 `MultiLabelBinarizer` 对“Research Areas”进行二元化。
    * 输出: `data/processed/multilabel_targets_research_areas.npy` (973, 97) 和 `research_areas_classes.pkl`
* **数据划分**: 将每种嵌入和对应标签划分为训练集(80%)和测试集(20%)。
    * 输出: `data/classification_data/<embedding_type>/{X_train, X_test, y_train, y_test}.npy`
* **模型实现**:
    * **1D CNN**: 对四种嵌入均进行了测试。
    * **LSTM (双向)**: 对BERT嵌入进行了测试。
* **评估**: 采用子集准确率、F1 Score (Micro, Macro, Weighted, Samples) 及分类报告，并进行了预测阈值优化。
* 主要模型及结果保存路径:
    * 模型: `models/<model_type>_<embedding_type>.h5`
    * 评估分数: `results/classification/accuracy_scores.txt`
    * 图表: `results/classification/` 目录下

### 3.4 主题分析与社区发现 (对应报告第7节)
* **输入数据**: 主要使用BERT嵌入或清洗后的文本。
* **聚类算法**:
    * **K-Means**: 基于BERT嵌入，通过轮廓系数分析选择K=4。
        * 输出: `results/clustering/kmeans_clusters_bert_k4.csv`, `kmeans_report_bert_k4.txt`
    * **层次聚类**: 基于BERT嵌入，使用Ward链接法，生成树状图。
        * 输出: `results/clustering/hierarchical_dendrogram_bert_ward.png`
* **LDA主题建模**:
    * 基于清洗后的文本，通过困惑度和主题一致性分析选择8个主题。
    * 输出: `models/lda_model.pkl`, `results/topic_modeling/lda_topics_k8.txt`, `lda_final_metrics_k8.txt`
    * **主题演化分析**: 分析8个LDA主题随文献发表年份的强度变化。
        * 输出: `results/topic_modeling/lda_topic_evolution_over_years.png`, `lda_topic_evolution_data.csv`
* **社区发现**:
    * 构建词共现网络，应用Louvain算法。
    * 输出: `results/community_detection/louvain_communities.txt`, `word_cooccurrence_graph.gexf`

## 4. 项目文件结构 

```bash
.
├── data/
│   ├── raw/
│   │   └── data.csv                     # 原始数据
│   ├── processed/
│   │   ├── cleaned_text.pkl
│   │   ├── tfidf_features.pkl
│   │   ├── tfidf_feature_names.pkl
│   │   ├── multilabel_targets_research_areas.npy
│   │   └── research_areas_classes.pkl
│   ├── embeddings/
│   │   ├── one_hot.npy
│   │   ├── word2vec_skipgram_embeddings.npy
│   │   ├── word2vec_cbow_embeddings.npy
│   │   └── bert_embeddings.npy
│   └── classification_data/
│       ├── one_hot/
│       ├── word2vec_skipgram/
│       ├── word2vec_cbow/
│       └── bert/
│           ├── X_train.npy, X_test.npy, y_train.npy, y_test.npy
├── models/                                # 保存训练好的模型
│   ├── cnn_one_hot.h5
│   ├── cnn_word2vec_skipgram.h5
│   ├── cnn_word2vec_cbow.h5
│   ├── cnn_bert.h5
│   ├── lstm_bert.h5
│   ├── word2vec_skipgram.model
│   ├── word2vec_cbow.model
│   └── lda_model.pkl
├── results/                               # 保存实验结果
│   ├── classification/
│   │   ├── accuracy_scores.txt
│   │   └── *.png (各种图表)
│   ├── clustering/
│   │   ├── kmeans_clusters_bert_k4.csv
│   │   ├── kmeans_report_bert_k4.txt
│   │   └── hierarchical_dendrogram_bert_ward.png
│   ├── topic_modeling/
│   │   ├── lda_tuning_metrics.png
│   │   ├── lda_topics_k8.txt
│   │   ├── lda_final_metrics_k8.txt
│   │   ├── lda_topic_evolution_over_years.png
│   │   └── lda_topic_evolution_data.csv
│   └── community_detection/
│       ├── louvain_communities.txt
│       └── word_cooccurrence_graph.gexf
├── code/                                  # 存放所有Python脚本 (或者直接在根目录)
│   ├── 1_preprocessing.py                 # (类似这样的命名)
│   ├── 2a_one_hot.py
│   ├── 2b_word2vec.py                     # (包含Skip-gram和CBOW)
│   ├── 2c_bert_embeddings.py
│   ├── 3a_label_preparation.py
│   ├── 3b_data_split.py
│   ├── 4a_cnn_model.py                    # (可以设计成一个通用脚本，通过参数传入嵌入类型)
│   ├── 4b_lstm_model.py                   # (同上)
│   ├── 5a_kmeans_tuning.py
│   ├── 5b_kmeans_final.py
│   ├── 5c_hierarchical_clustering.py
│   ├── 6a_lda_tuning.py
│   ├── 6b_lda_final.py
│   ├── 6c_topic_evolution.py
│   └── 7_community_detection.py
├── 文本挖掘实验报告.docx                     # 您的最终实验报告
└── README.md                              # 本文件
```
## 5. 环境配置与依赖

* Python 3.12.10
* 主要依赖库:
    * pandas
    * numpy
    * nltk
    * scikit-learn
    * gensim
    * tensorflow (或 keras 单独安装)
    * torch (用于transformers)
    * transformers
    * networkx
    * python-louvain
    * matplotlib
    * seaborn

建议创建一个虚拟环境，并通过 `pip install -r requirements.txt` 安装依赖。

## 6. 如何运行 

1.  确保原始数据 `data/raw/data.csv` 已放置正确。
2.  依次执行 `code/` 目录下的脚本，例如：
    * 运行数据预处理脚本 (如 `1_preprocessing.py`)。
    * 运行文本表示生成脚本 (如 `2a_one_hot.py`, `2b_word2vec.py`, `2c_bert_embeddings.py`)。
    * 运行标签准备和数据划分脚本 (如 `3a_label_preparation.py`, `3b_data_split.py`)。
    * 运行分类模型训练与评估脚本 (如 `4a_cnn_model.py` -- 可能需要修改脚本以遍历不同嵌入或作为参数传入)。
    * 运行主题分析和社区发现脚本。
    * *(请您根据您的实际脚本执行顺序和方式进行详细说明)*
3.  脚本会自动在 `models/` 和 `results/` 目录下生成相应的模型、数据文件和图表。

## 7. 主要实验结论 (简述)

* 在文本分类任务中，LSTM + BERT 嵌入的组合在F1 Micro分数上表现最佳 (约0.4302，最优阈值0.20)，优于各种CNN组合。
* 语义嵌入 (Word2Vec, BERT) 显著优于One-Hot编码。
* 数据集的样本量、类别数量及不均衡性是影响模型性能的主要限制因素。
* K-Means (K=4, BERT嵌入) 和层次聚类揭示了数据中一些主题倾向，但簇结构不强 (轮廓系数较低)。
* LDA模型 (8个主题) 提取了具有一定可解释性的主题，主题演化分析展示了其年度变化趋势。
* Louvain社区发现 (基于词共现) 发现了4个功能性词语社区，但模块度较低。

更详细的分析已写入 `文本挖掘实验报告.docx`。

## 8. 未来工作与改进方向 (对应报告8.2节)

* 文本分类性能受多重因素制约，数据本身的特性是制约当前分类性能的主要瓶颈。尽管采用了较先进的表示方法和模型，但由于数据集样本量（973篇）相对于类别数量（97个）和标签稀疏性、不均衡性而言存在显著不足，所有模型在宏平均F1分数和许多低频类别的识别上均表现有限，子集准确率也普遍不高。

## 9. 致谢 (Optional)

- 致敬伟大的`Gemini-2.5-Pro-Preview(0506)`、`Claude-3.7-Sonnet`、`GPT-4-Sonnet`和`Deepseel-V3-0324`，没有你们就没有这份作业的完成。
