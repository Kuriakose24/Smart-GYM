import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load posture model
model = joblib.load("pushup_posture_model.pkl")

# Load pose detector
base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.PoseLandmarker.create_from_options(options)

# Load image
image = cv2.imread("test_pushup.png")

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

result = detector.detect(mp_image)

if result.pose_landmarks:

    landmarks = result.pose_landmarks[0]

    def get_point(i):
        return [landmarks[i].x, landmarks[i].y]

    def calculate_angle(a,b,c):
        a=np.array(a); b=np.array(b); c=np.array(c)
        radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
        angle=np.abs(radians*180/np.pi)
        if angle>180: angle=360-angle
        return angle

    shoulder = get_point(11)
    elbow = get_point(13)
    wrist = get_point(15)
    hip = get_point(23)
    knee = get_point(25)
    ankle = get_point(27)

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    back_angle = calculate_angle(shoulder, hip, knee)

    hip_angle = calculate_angle(shoulder, hip, knee)

    knee_angle = calculate_angle(hip, knee, ankle)

    print("Angles:", elbow_angle, back_angle, hip_angle, knee_angle)

    sample = pd.DataFrame(
        [[elbow_angle,back_angle,hip_angle,knee_angle]],
        columns=["elbow_angle","back_angle","hip_angle","knee_angle"]
    )

    prediction = model.predict(sample)

    print("Posture:", prediction[0])

else:
    print("No pose detected")