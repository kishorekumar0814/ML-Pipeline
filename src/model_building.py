import os
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier

# ✅ Get project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Path to features
features_path = os.path.join(base_dir, "data", "features")
train_path = os.path.join(features_path, "train_bow.csv")

# ✅ Read training data
train_data = pd.read_csv(train_path)

# ✅ Separate features and labels
X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# ✅ Define and train the Gradient Boosting model
clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# ✅ Save the trained model inside models/ folder
models_path = os.path.join(base_dir, "models")
os.makedirs(models_path, exist_ok=True)

model_file = os.path.join(models_path, "model.pkl")
with open(model_file, "wb") as f:
    pickle.dump(clf, f)

print("\n✅ Model training completed!")
print(f"📌 Model saved at: {model_file}")
