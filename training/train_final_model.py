import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load final dataset
df = pd.read_csv("pushup_final_dataset.csv")

X = df[["elbow_angle","back_angle","hip_angle","knee_angle"]]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nFeature Importance:")
for f, i in zip(X.columns, model.feature_importances_):
    print(f, ":", round(i,3))

# Save model
joblib.dump(model, "pushup_final_model.pkl")

print("\nModel saved as pushup_final_model.pkl")