"""Train the Iris classifier and save it to model.pkl (+ model_info.json).
Run once:  python train_model.py
"""
import json
from datetime import datetime, timezone

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
]).fit(X_train, y_train)

info = {
    "model": "StandardScaler + LogisticRegression",
    "version": "1.0",
    "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    "train_accuracy": round(model.score(X_train, y_train), 4),
    "test_accuracy": round(model.score(X_test, y_test), 4),
    "n_train": len(X_train),
    "n_test": len(X_test),
}
joblib.dump(model, "model.pkl")
with open("model_info.json", "w") as f:
    json.dump(info, f, indent=2)
print("Saved model.pkl and model_info.json:", info)
