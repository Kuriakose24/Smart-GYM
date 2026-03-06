import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("pushup_labeled_dataset.csv")

# Features
X = df[["elbow_angle","back_angle","hip_angle","knee_angle"]]

# Labels
y = df["label"]

# Train-test split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100)

# Train
model.fit(X_train,y_train)

# Accuracy
accuracy = model.score(X_test,y_test)
print("Model Accuracy:",accuracy)

# Save model
joblib.dump(model,"pushup_posture_model.pkl")

print("Model saved as pushup_posture_model.pkl")