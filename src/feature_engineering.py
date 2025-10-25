import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ✅ Get project root directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Paths to processed data
processed_data_path = os.path.join(base_dir, "data", "processed")
train_path = os.path.join(processed_data_path, "train_processed.csv")
test_path = os.path.join(processed_data_path, "test_processed.csv")

# ✅ Read processed train & test data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Fill missing values
train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# ✅ Separate features and labels
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

X_test = test_data['content'].values
y_test = test_data['sentiment'].values

# ✅ Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=50)
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# ✅ Convert to DataFrame and add label column
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# ✅ Save features inside data/features
features_path = os.path.join(base_dir, "data", "features")
os.makedirs(features_path, exist_ok=True)

train_df.to_csv(os.path.join(features_path, "train_bow.csv"), index=False)
test_df.to_csv(os.path.join(features_path, "test_bow.csv"), index=False)

print("\n✅ Feature engineering completed!")
print(f"📌 Features saved at: {features_path}")
