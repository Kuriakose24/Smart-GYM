import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# -----------------------------
# Load Balanced Dataset
# -----------------------------

df = pd.read_csv("datasets/squat_clean_dataset.csv")

print("Dataset Loaded")
print(df.head())

print("\nLabel Distribution:")
print(df["label"].value_counts())


# -----------------------------
# Features and Labels
# -----------------------------

X = df[[
    "knee_angle",
    "hip_angle",
    "back_angle"
]]

y = df["label"]


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluation
# -----------------------------

pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))


# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "models/squat_final_model.pkl")

print("\nModel saved → models/squat_final_model.pkl")