import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Define the file path
file_path = r'D:\Codes\textmining\data\raw\data.csv' # Use raw string for Windows paths

# IMPORTANT: Specify the path to a font file that supports Chinese characters
# Common fonts include SimHei, Microsoft YaHei, Noto Sans CJK, etc.
# Example for Windows: font_path = 'C:/Windows/Fonts/simhei.ttf'
# Example for macOS: font_path = '/Library/Fonts/Songti.ttc'
# Example for Linux (if you've installed it): '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
# Please replace this with a valid path on your system.
# If you don't have a specific font, you might need to download one (e.g., Noto Sans CJK)
# and provide the path to the .ttf or .otf file.
font_path = None # <<< --- !!! SET THIS TO YOUR FONT FILE PATH !!! --- >>>
# As a fallback if no font_path is set, it will try default fonts which might not render CJK.

try:
    # Load the CSV file
    df = pd.read_csv(file_path)

    # --- Initial Data Inspection ---
    print("--- First 5 rows of the DataFrame (to check 'Article Title' column): ---")
    print(df[['Article Title']].head())
    print(f"\n--- Checking for 'Article Title' column and its data type ---")
    if 'Article Title' in df.columns:
        print(f"Data type of 'Article Title': {df['Article Title'].dtype}")
        # Handle potential missing values in 'Article Title'
        # Convert NaN to empty string, and ensure all are strings
        titles_text = ' '.join(df['Article Title'].dropna().astype(str).tolist())

        if not titles_text.strip():
            print("\nError: The 'Article Title' column is empty or contains only missing values after processing.")
        else:
            print(f"\nSuccessfully extracted text from 'Article Title'. Total characters: {len(titles_text)}")

            # Define some common stopwords (you can add more specific to your domain)
            stopwords = set(STOPWORDS)
            stopwords.update(["et", "al", "review", "study", "research", "analysis", "based", "using", "article"])


            # --- Generate Word Cloud ---
            print("\n--- Generating word cloud... ---")
            # Attempt to use the user-provided font_path
            # If font_path is not set or invalid, WordCloud might use a default font
            # which may not support Chinese characters.
            if font_path:
                wordcloud = WordCloud(width=800, height=400,
                                      background_color='white',
                                      stopwords=stopwords,
                                      font_path=font_path, # Specify font path for CJK characters
                                      collocations=False, # Avoids treating common word pairs as single entities
                                      min_font_size=10).generate(titles_text)
            else:
                print("\nWarning: `font_path` is not set. Chinese characters might not display correctly.")
                print("Please set the `font_path` variable in the script to a valid .ttf or .otf font file that supports Chinese.")
                wordcloud = WordCloud(width=800, height=400,
                                      background_color='white',
                                      stopwords=stopwords,
                                      collocations=False,
                                      min_font_size=10).generate(titles_text)


            # --- Display the Word Cloud ---
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off") # Don't show axes
            plt.tight_layout(pad=0)

            # Save the word cloud image
            wordcloud_filename = 'article_title_wordcloud.png'
            plt.savefig(wordcloud_filename)
            print(f"\nWord cloud image saved as '{wordcloud_filename}'")

            plt.show()
    else:
        print("\nError: 'Article Title' column not found in the CSV file.")
        print("Available columns are:")
        print(df.columns.tolist())

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
except ImportError:
    print("Error: The 'wordcloud' library is not installed. Please install it by running: pip install wordcloud")
except RuntimeError as e:
    if "Could not find a font" in str(e) and not font_path:
        print(f"RuntimeError: {e}")
        print("This often means a suitable font for all characters in your titles was not found by the WordCloud library.")
        print("Please ensure you set the 'font_path' variable in the script to a valid .ttf or .otf font file that supports all characters in your titles (especially CJK characters if present).")
    else:
        print(f"An error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")