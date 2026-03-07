import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


# ----------------------------
# Load dataset
# ----------------------------

pd.read_csv("datasets/pushup_image_dataset_balanced.csv")

X = df[["elbow_angle","back_angle","hip_angle","knee_angle"]]
y = df["label"]


# ----------------------------
# Split dataset
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ----------------------------
# Train model
# ----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)


# ----------------------------
# Evaluate model
# ----------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# ----------------------------
# Feature Importance
# ----------------------------

print("\nFeature Importance:")

features = X.columns
importances = model.feature_importances_

for f, i in zip(features, importances):
    print(f"{f}: {round(i,3)}")


# ----------------------------
# Save model
# ----------------------------

joblib.dump(model, "pushup_posture_model.pkl")

print("\nModel saved as pushup_posture_model.pkl")