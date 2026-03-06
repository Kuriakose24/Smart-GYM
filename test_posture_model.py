import joblib
import numpy as np

# Load trained model
model = joblib.load("pushup_posture_model.pkl")

print("Model loaded successfully\n")

# Test samples
test_samples = [

    [80,170,165,150],   # correct pushup
    [95,175,170,160],   # correct pushup
    [160,120,110,170],  # incorrect back bending
    [150,100,90,170],   # incorrect posture
]

for sample in test_samples:

    prediction = model.predict([sample])

    print("Angles:", sample)
    print("Prediction:", prediction[0])
    print("---------------")