本项目为课程《统计机器学习与文本挖掘》上机实验对应材料。

# 文件存储结构

```plain
silver_economy_experiment/
├── README.md
├── data/
│   ├── raw/                     # 原始数据 
│   │   └── data.csv    # 用户提供的原始CSV文件
│   ├── processed/               # 预处理后的数据 
│   │   ├── cleaned_text.pkl      # 清洗后的文本数据 
│   │   └── tfidf_features.pkl    # TF-IDF特征矩阵 
│   └── embeddings/              # 文本表示结果 
│       ├── one_hot.npy 
│       ├── word2vec_embeddings.npy 
│       └── bert_embeddings.npy  
├── code/
│   ├── 1_preprocessing.py        # 数据预处理代码 
│   ├── 2_representation.py       # 文本表示代码 
│   ├── 3_classification.py       # 文本分类代码 
│   └── 4_topic_analysis.py       # 聚类与主题分析代码 
├── models/                      # 保存训练好的模型 
│   ├── cnn_model.h5 
│   ├── lstm_model.h5 
│   └── lda_model.pkl 
├── results/                     # 实验输出
│   ├── classification/
│   │   ├── accuracy_scores.txt   # 分类评估结果 
│   │   └── confusion_matrix.png 
│   └── clustering/
│       ├── kmeans_clusters.csv  
│       └── lda_topics.txt  
└── docs/                        # 实验报告相关 
    └── report.md  
```

# **具体工作流程**

实验各环节的操作步骤及文件生成逻辑：

## **（1）数据预处理**

* **目标** ：清洗文本、提取特征词
* **核心步骤** ：

1. **分词** ：使用NLTK或Jieba（中文）对摘要字段分词。
2. **词形还原** ：英文用NLTK的 `WordNetLemmatizer`，中文需自定义词典。
3. **停用词处理** ：移除通用停用词（如"the", "and"）和领域无关低频词。
4. **TF-IDF筛选** ：计算词频-逆文档频率，保留前500-1000个关键词。

* **输出文件** ：`processed/cleaned_text.pkl` （清洗后文本）、`processed/tfidf_features.pkl` （TF-IDF特征矩阵）

## **（2）文本表示**

* **方法对比** ：
* **One-Hot** ：适用于短文本，稀疏高维（`embeddings/one_hot.npy` ）
* **Word2Vec** ：捕捉语义关联，需预训练或从头训练（`embeddings/word2vec_embeddings.npy` ）
* **BERT** ：上下文敏感表示，推荐HuggingFace的 `bert-base-uncased`（`embeddings/bert_embeddings.npy` ）
* **代码逻辑** ：不同表示方法分开保存，便于后续模型调用。

## **（3）文本分类**

* **标签选择** ：基于WOS学科分类（如 `Economics`, `Gerontology`），建议采用二级分类体系。
* **模型实现** ：
* **CNN** ：用Keras构建1D卷积层（`models/cnn_model.h5`）
* **LSTM/RNN** ：处理序列依赖（`models/lstm_model.h5`）
* **评估指标** ：保存准确率、F1-score到 `results/classification/accuracy_scores.txt` ，混淆矩阵可视化（PNG格式）。

## **（4）主题分析**

* **聚类算法** ：
* **K-Means** ：划分核心主题（`results/clustering/kmeans_clusters.csv` ）
* **层次聚类** ：树状图可视化
* **LDA** ：生成主题-词分布（`models/lda_model.pkl` ）
* **社区发现** ：使用Louvain算法结合语义网络（需构建共现矩阵）
* **评估指标** ：轮廓系数（K-Means）、困惑度（LDA）
