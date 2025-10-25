import os
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ✅ Get project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Paths
model_path = os.path.join(base_dir, "models", "model.pkl")
features_path = os.path.join(base_dir, "data", "features", "test_bow.csv")
reports_path = os.path.join(base_dir, "reports")
os.makedirs(reports_path, exist_ok=True)

metrics_file = os.path.join(reports_path, "metrics.json")

# ✅ Load model and test data
clf = pickle.load(open(model_path, 'rb'))
test_data = pd.read_csv(features_path)

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# ✅ Make predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# ✅ Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "auc": auc
}

# ✅ Save metrics to reports/metrics.json
with open(metrics_file, 'w') as f:
    json.dump(metrics_dict, f, indent=4)

print("\n✅ Model evaluation completed!")
print(f"📌 Metrics saved at: {metrics_file}")
print(metrics_dict)
