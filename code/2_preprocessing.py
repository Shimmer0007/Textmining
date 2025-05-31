import pandas as pd
import nltk
import re
import pickle
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords # NLTK的停用词库
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 检查并下载NLTK资源 ---
nltk_resources = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4', # wordnet的依赖
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
}

for resource_name, resource_path in nltk_resources.items():
    try:
        nltk.data.find(resource_path)
        print(f"NLTK资源 '{resource_name}' 已存在。")
    except LookupError:
        print(f"NLTK资源 '{resource_name}' 未找到。正在尝试下载...")
        try:
            nltk.download(resource_name, quiet=False)
            print(f"NLTK资源 '{resource_name}' 下载完成。")
            # 对于wordnet，可能需要显式检查omw-1.4是否也已作为依赖下载
            if resource_name == 'wordnet':
                 nltk.data.find('corpora/omw-1.4') # 再次检查，确保omw-1.4也存在
                 print(f"NLTK资源 'omw-1.4' (wordnet的依赖) 已确认。")
        except Exception as e:
            print(f"下载NLTK资源 '{resource_name}' 失败: {e}")
            print("请尝试手动在Python解释器中运行以下命令以下载:")
            print(f">>> import nltk")
            print(f">>> nltk.download('{resource_name}')")
            if resource_name == 'wordnet':
                print("对于 'wordnet', 也请确保 'omw-1.4' 已下载:")
                print(f">>> nltk.download('omw-1.4')")
            exit() # 如果关键资源下载失败，则退出


# --- 创建输出目录 (如果不存在) ---
output_dir = r'D:\Codes\textmining\data\processed' # 输出文件将保存在指定的绝对路径下
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建目录: {output_dir}")
else:
    print(f"目录 {output_dir} 已存在。")

# --- 1. 加载数据 ---
print("正在加载数据文件...")
file_path = r'D:\Codes\textmining\data\raw\data.csv' # 您提供的完整路径

try:
    df = pd.read_csv(file_path)
    print(f"成功加载 {file_path}")
    print(f"数据框的形状: {df.shape}")
    
    if 'Abstract' not in df.columns:
        print("错误：在CSV文件中未找到 'Abstract' 列。")
        print(f"可用的列有: {df.columns.tolist()}")
        abstracts = pd.Series([], dtype=str) 
    else:
        abstracts = df['Abstract'].astype(str).fillna('') 
    
    print(f"待处理的摘要数量: {len(abstracts)}")
    if not abstracts.empty:
        print("前几条摘要 (head):")
        print(abstracts.head())

except FileNotFoundError:
    print(f"错误: 文件 {file_path} 未找到。请确保文件路径正确。")
    exit() 
except Exception as e:
    print(f"加载CSV文件时发生错误: {e}")
    exit() 

# --- 2. 文本预处理函数 ---
lemmatizer = WordNetLemmatizer()
# 使用NLTK的英文停用词列表
try:
    stop_words_set = set(stopwords.words('english'))
except LookupError:
    print("错误: NLTK的 'stopwords' 资源仍未正确加载，即使下载尝试过。请检查NLTK安装和资源。")
    exit()


def get_wordnet_pos(word):
    """将NLTK的POS tag转换为WordNetLemmatizer使用的格式"""
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except LookupError:
        print("错误: NLTK的 'averaged_perceptron_tagger' 资源未加载。词形还原可能不准确。")
        # 如果tagger缺失，默认返回名词，或者可以考虑退出
        return nltk.corpus.wordnet.NOUN 
    except IndexError: # 处理空字符串或无法标记的词
        return nltk.corpus.wordnet.NOUN


    tag_dict = {"J": nltk.corpus.wordnet.ADJ,
                "N": nltk.corpus.wordnet.NOUN,
                "V": nltk.corpus.wordnet.VERB,
                "R": nltk.corpus.wordnet.ADV}
    return tag_dict.get(tag, nltk.corpus.wordnet.NOUN)

def preprocess_text(text):
    if pd.isna(text) or text.strip() == "":
        return []
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S*@\S*\s?', '', text)
    text = re.sub(r'[^a-z\s]', '', text) # 只保留小写字母和空格
    
    try:
        tokens = word_tokenize(text)
    except LookupError:
        print("错误: NLTK的 'punkt' 资源未加载。无法进行分词。")
        return [] # 或者可以考虑退出

    processed_tokens = [
        lemmatizer.lemmatize(w, get_wordnet_pos(w)) 
        for w in tokens 
        if w.isalpha() and w not in stop_words_set and len(w) > 2
    ]
    return processed_tokens

# --- 3. 应用预处理 ---
print("\n开始文本预处理...")
if abstracts.empty:
    print("未找到可处理的摘要 (例如 'Abstract' 列缺失或为空)。")
    cleaned_texts_tokens = []
else:
    # 使用列表推导处理所有摘要
    cleaned_texts_tokens = [preprocess_text(abstract) for abstract in abstracts]

print("文本预处理完成。")
print(f"清洗后的文本数量: {len(cleaned_texts_tokens)}")

if cleaned_texts_tokens:
    # 查找第一个非空的处理后摘要进行打印
    first_processed_sample = next((item for item in cleaned_texts_tokens if item), None)
    if first_processed_sample:
        print("第一个有效摘要清洗后的词元示例 (前20个词元):")
        print(first_processed_sample[:20])
    else:
        print("所有摘要在清洗后均为空。")
else:
    print("没有摘要被处理，或所有摘要在清洗后均为空。")

# 准备用于TF-IDF的文档 (过滤掉清洗后为空的文档)
processed_docs_for_tfidf = [" ".join(tokens) for tokens in cleaned_texts_tokens if tokens]

if not processed_docs_for_tfidf:
    print("错误：预处理后没有剩余文档，无法计算TF-IDF。")
    
    cleaned_text_path = os.path.join(output_dir, 'cleaned_text.pkl')
    with open(cleaned_text_path, 'wb') as f:
        pickle.dump(cleaned_texts_tokens, f)
    print(f"已将（可能为空的）清洗后文本保存至 {cleaned_text_path}")
    
    tfidf_features_path = os.path.join(output_dir, 'tfidf_features.pkl')
    tfidf_vocab_path = os.path.join(output_dir, 'tfidf_feature_names.pkl')
    if os.path.exists(tfidf_features_path): os.remove(tfidf_features_path)
    if os.path.exists(tfidf_vocab_path): os.remove(tfidf_vocab_path)
    print("由于预处理后无有效文档内容，TF-IDF特征生成已跳过。")

else:
    print(f"\n用于TF-IDF的文档数量: {len(processed_docs_for_tfidf)}")
    print("正在计算TF-IDF特征...")
    tfidf_vectorizer = TfidfVectorizer(max_features=800, min_df=3, max_df=0.9, stop_words='english') 
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs_for_tfidf)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    print("TF-IDF计算完成。")
    print(f"TF-IDF矩阵的形状: {tfidf_matrix.shape if hasattr(tfidf_matrix, 'shape') else 'N/A'}") 
    print(f"选定的特征数量: {len(feature_names)}")
    if feature_names.size > 0:
         print(f"部分特征名 (前10个): {feature_names[:10].tolist()}") # .tolist() for better printing
    else:
        print("TF-IDF未选择任何特征。")

    # --- 5. 保存输出 ---
    cleaned_text_path = os.path.join(output_dir, 'cleaned_text.pkl')
    with open(cleaned_text_path, 'wb') as f:
        pickle.dump(cleaned_texts_tokens, f)
    print(f"清洗后的文本 (词元列表的列表) 已保存至 {cleaned_text_path}")

    tfidf_features_path = os.path.join(output_dir, 'tfidf_features.pkl')
    with open(tfidf_features_path, 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"TF-IDF特征矩阵已保存至 {tfidf_features_path}")

    tfidf_vocab_path = os.path.join(output_dir, 'tfidf_feature_names.pkl')
    with open(tfidf_vocab_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"TF-IDF特征名已保存至 {tfidf_vocab_path}")

print("\n数据预处理脚本执行完毕。")

# --- 验证 (可选：尝试加载已保存的文件) ---
print("\n正在验证已保存的文件...")
try:
    with open(os.path.join(output_dir, 'cleaned_text.pkl'), 'rb') as f:
        loaded_cleaned_text = pickle.load(f)
    print(f"成功加载 cleaned_text.pkl。条目数量: {len(loaded_cleaned_text)}")
    if loaded_cleaned_text:
        first_loaded_sample = next((item for item in loaded_cleaned_text if item), None)
        if first_loaded_sample:
            print(f"  加载的第一个有效条目示例 (前10个词元): {first_loaded_sample[:10]}")
        else:
            print("  加载的cleaned_text.pkl中所有条目均为空列表。")


    if os.path.exists(os.path.join(output_dir, 'tfidf_features.pkl')):
        with open(os.path.join(output_dir, 'tfidf_features.pkl'), 'rb') as f:
            loaded_tfidf_matrix = pickle.load(f)
        print(f"成功加载 tfidf_features.pkl。形状: {loaded_tfidf_matrix.shape if hasattr(loaded_tfidf_matrix, 'shape') else 'N/A (可能为None或空)'}")
    else:
        print("tfidf_features.pkl 未创建 (预期情况，若无文档用于TF-IDF)。")
        
    if os.path.exists(os.path.join(output_dir, 'tfidf_feature_names.pkl')):
        with open(os.path.join(output_dir, 'tfidf_feature_names.pkl'), 'rb') as f:
            loaded_tfidf_vocab = pickle.load(f)
        print(f"成功加载 tfidf_feature_names.pkl。特征数量: {len(loaded_tfidf_vocab)}")
        if loaded_tfidf_vocab.size > 0 : # Check if numpy array is not empty
            print(f"  特征示例 (前10个): {list(loaded_tfidf_vocab[:10])}")
    else:
        print("tfidf_feature_names.pkl 未创建 (预期情况，若TF-IDF无特征)。")

except FileNotFoundError as fnf_e:
    print(f"验证错误：文件未找到。{fnf_e}")
except Exception as e:
    print(f"加载pickle文件进行验证时发生错误: {e}")