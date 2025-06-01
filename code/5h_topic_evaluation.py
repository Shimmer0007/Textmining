import numpy as np
import pandas as pd
import os
import pickle
import gensim
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
gensim_logger = logging.getLogger("gensim")
gens_logger_level = getattr(logging, os.environ.get("GENSIM_LOGGING", "WARNING").upper(), None)
if gens_logger_level is None:
    gens_logger_level = logging.WARNING
gensim_logger.setLevel(gens_logger_level)

# --- Configure Chinese font for plotting ---
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- Define Paths ---
base_data_path = r'D:\Codes\textmining\data'
cleaned_text_path = os.path.join(base_data_path, 'processed', 'cleaned_text.pkl')
raw_data_path = os.path.join(base_data_path, 'raw', 'data.csv') # For publication year

models_dir = os.path.join(base_data_path, '..', 'models')
lda_model_path = os.path.join(models_dir, 'lda_model.pkl')

output_results_dir = os.path.join(base_data_path, '..', 'results', 'topic_modeling')
topic_evolution_plot_path = os.path.join(output_results_dir, 'lda_topic_evolution_over_years.png')
topic_evolution_data_path = os.path.join(output_results_dir, 'lda_topic_evolution_data.csv')


def main():
    # --- Create output directory if it doesn't exist ---
    if not os.path.exists(output_results_dir):
        os.makedirs(output_results_dir)
        print(f"Directory created: {output_results_dir}")

    # --- 1. Load Data ---
    print(f"\n--- Loading trained LDA model from '{lda_model_path}' ---")
    if not os.path.exists(lda_model_path):
        print(f"Error: LDA model file '{lda_model_path}' not found.")
        return
    try:
        lda_model = LdaModel.load(lda_model_path)
        dictionary = lda_model.id2word # The dictionary is part of the saved model
        num_topics = lda_model.num_topics
        print(f"LDA model loaded successfully with {num_topics} topics.")
    except Exception as e:
        print(f"Error loading LDA model: {e}")
        return

    print(f"\n--- Loading cleaned text data from '{cleaned_text_path}' ---")
    if not os.path.exists(cleaned_text_path):
        print(f"Error: Cleaned text file '{cleaned_text_path}' not found.")
        return
    try:
        with open(cleaned_text_path, 'rb') as f:
            documents = pickle.load(f)
        documents = [doc for doc in documents if doc] # Ensure non-empty documents
        print(f"Successfully loaded {len(documents)} non-empty documents.")
    except Exception as e:
        print(f"Error loading cleaned text data: {e}")
        return

    print(f"\n--- Loading publication years from '{raw_data_path}' ---")
    if not os.path.exists(raw_data_path):
        print(f"Error: Original data file '{raw_data_path}' not found.")
        return
    try:
        df_original = pd.read_csv(raw_data_path, usecols=['Publication Year'])
        # Ensure 'Publication Year' is numeric and handle errors/NaNs
        df_original['Publication Year'] = pd.to_numeric(df_original['Publication Year'], errors='coerce')
        df_original.dropna(subset=['Publication Year'], inplace=True) # Remove rows with invalid years
        df_original['Publication Year'] = df_original['Publication Year'].astype(int)
        
        # Assuming the order of documents in cleaned_text.pkl matches the original CSV.
        # If not, an explicit join based on a document ID would be needed.
        # For this script, we assume the lengths will match after filtering.
        # We need to filter `documents` to match `df_original` after dropping NaNs in year.
        
        # Create an initial index for documents before year filtering
        original_indices = list(range(len(pickle.load(open(cleaned_text_path, 'rb')))))
        
        # Filter original_indices and documents based on valid years
        valid_year_indices = df_original.index # Indices of rows with valid years
        
        # Select documents and their original indices that correspond to valid years
        # This assumes df_original maintains original row order from the full CSV when 'Publication Year' is first read.
        
        # Re-filter documents based on rows that had valid years in df_original
        # This alignment can be tricky. A safer way is to load an ID column as well if available.
        # For now, if lengths differ significantly, it will be problematic.
        
        # Let's assume the `documents` list corresponds to the original CSV rows.
        # We need to align `documents` with `df_original` which has NaN years dropped.
        publication_years = df_original['Publication Year'].values
        
        if len(documents) != len(df_original): # Check if original CSV and cleaned_text have different initial lengths
            print(f"Warning: Initial document count ({len(pickle.load(open(cleaned_text_path, 'rb')))}) "
                  f"differs from rows with 'Publication Year' in CSV ({len(pd.read_csv(raw_data_path, usecols=['Publication Year']))}).")
            print("Attempting alignment based on non-empty documents and valid years.")
            
            # This is a simplified alignment: assumes documents correspond to rows in order,
            # and we filter `documents` if its corresponding row in `df_original` (before dropping NaNs) had a NaN year.
            # A more robust approach would use document IDs if available.
            
            # Let's reload df_original with an index to track which rows were dropped
            df_full_with_year = pd.read_csv(raw_data_path, usecols=['Publication Year'])
            df_full_with_year['original_index'] = df_full_with_year.index
            df_full_with_year.dropna(subset=['Publication Year'], inplace=True)
            valid_original_indices = df_full_with_year['original_index'].tolist()
            
            temp_documents = pickle.load(open(cleaned_text_path, 'rb'))
            documents = [temp_documents[i] for i in valid_original_indices if i < len(temp_documents)]
            documents = [doc for doc in documents if doc] # Filter empty again after selection
            
            publication_years = pd.to_numeric(df_full_with_year['Publication Year'], errors='coerce').astype(int).values
            
            if len(documents) != len(publication_years):
                print(f"Error: Misalignment after attempting to match documents with publication years. "
                      f"Docs: {len(documents)}, Years: {len(publication_years)}. Cannot proceed.")
                return

        print(f"Successfully loaded {len(publication_years)} publication years, aligned with {len(documents)} documents.")
        
    except Exception as e:
        print(f"Error loading or processing publication years: {e}")
        return

    # --- 2. Recreate Corpus and Get Topic Distributions ---
    print("\n--- Recreating BoW corpus and getting topic distributions for documents ---")
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    doc_topic_distributions = []
    for i, bow_doc in enumerate(corpus):
        if not bow_doc: # Handle empty documents after BoW conversion
            doc_topic_distributions.append([0.0] * num_topics)
            continue
        # Get topic distribution, ensure all topics are present with 0 prob if not assigned
        topic_dist = lda_model.get_document_topics(bow_doc, minimum_probability=0.0)
        dist_array = np.zeros(num_topics)
        for topic_id, prob in topic_dist:
            dist_array[topic_id] = prob
        doc_topic_distributions.append(dist_array)
    
    doc_topic_distributions_df = pd.DataFrame(doc_topic_distributions, columns=[f'Topic_{i}' for i in range(num_topics)])
    doc_topic_distributions_df['Publication_Year'] = publication_years
    
    if doc_topic_distributions_df.empty:
        print("Error: No topic distributions could be generated.")
        return

    # --- 3. Calculate Average Topic Intensity per Year ---
    print("\n--- Calculating average topic intensity per year ---")
    yearly_topic_intensity = doc_topic_distributions_df.groupby('Publication_Year').mean()
    
    # Ensure all years in the original range are present, fill missing with NaN or 0 if preferred
    if not yearly_topic_intensity.empty:
        min_year, max_year = int(yearly_topic_intensity.index.min()), int(yearly_topic_intensity.index.max())
        all_years_range = pd.RangeIndex(start=min_year, stop=max_year + 1, name='Publication_Year')
        yearly_topic_intensity = yearly_topic_intensity.reindex(all_years_range).fillna(0) # Fill missing years with 0 intensity
    
    print("Yearly topic intensity data (head):")
    print(yearly_topic_intensity.head())
    yearly_topic_intensity.to_csv(topic_evolution_data_path)
    print(f"Yearly topic intensity data saved to: {topic_evolution_data_path}")


    # --- 4. Visualize Topic Evolution ---
    if yearly_topic_intensity.empty:
        print("No data to plot for topic evolution.")
        return
        
    print("\n--- Plotting topic evolution over years ---")
    plt.figure(figsize=(15, 8))
    for i in range(num_topics):
        plt.plot(yearly_topic_intensity.index, yearly_topic_intensity[f'Topic_{i}'], label=f'主题 #{i}', linewidth=2)
        
    plt.xlabel('发表年份 (Publication Year)')
    plt.ylabel('平均主题强度 (Average Topic Intensity)')
    plt.title(f'LDA ({num_topics}个主题) 主题强度随年份演化趋势', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    try:
        plt.savefig(topic_evolution_plot_path)
        print(f"Topic evolution plot saved to: {topic_evolution_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show(block=False)

    print("\n--- 主题演化分析完成 ---")

if __name__ == '__main__':
    main()