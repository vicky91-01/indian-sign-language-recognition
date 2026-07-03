import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------- LOAD DATA ----------
print("Loading your custom dataset...")
df = pd.read_csv("isl_landmarks.csv")
print(f"Loaded {len(df)} samples")

print("\nSamples per letter:")
print(df["label"].value_counts())

if len(df) < 50:
    print("\nWARNING: Very little data. Consider recording more samples per letter (aim for 100+ each) before trusting results.")

# ---------- PREPARE FEATURES ----------
feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols]
y = df["label"]

# ---------- TRAIN/TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------- TRAIN MODEL ----------
print("\nTraining Random Forest classifier...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("Training complete.")

# ---------- EVALUATE ----------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed classification report:")
print(classification_report(y_test, y_pred))

# ---------- SAVE MODEL ----------
MODEL_FILE = "isl_model_custom.pkl"
joblib.dump(model, MODEL_FILE)
print(f"\nModel saved to: {MODEL_FILE}")

FEATURES_FILE = "feature_columns_custom.pkl"
joblib.dump(feature_cols, FEATURES_FILE)
print(f"Feature column order saved to: {FEATURES_FILE}")

print("\nDone! Use isl_model_custom.pkl with your webcam for real-time predictions.")