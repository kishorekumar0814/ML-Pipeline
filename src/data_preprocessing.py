import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ✅ Go to project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_data_path = os.path.join(base_dir, "data", "raw")

# ✅ Read raw train & test data
train_data = pd.read_csv(os.path.join(raw_data_path, "train.csv"))
test_data = pd.read_csv(os.path.join(raw_data_path, "test.csv"))

# ✅ Download NLP resources
nltk.download('wordnet')
nltk.download('stopwords')

# ✅ Text cleaning functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])

def lower_case(text):
    return " ".join([y.lower() for y in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

# ✅ Apply preprocessing
train_processed = normalize_text(train_data)
test_processed = normalize_text(test_data)

# ✅ Save processed data inside data/processed
processed_data_path = os.path.join(base_dir, "data", "processed")
os.makedirs(processed_data_path, exist_ok=True)

train_processed.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
test_processed.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)

print("\n✅ Data Preprocessing Completed Successfully!")
print(f"📌 Processed files saved at: {processed_data_path}")
