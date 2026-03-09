import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd

model = joblib.load("models/squat_final_model.pkl")

base_options = python.BaseOptions(
    model_asset_path="models/pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.PoseLandmarker.create_from_options(options)


def calculate_angle(a,b,c):

    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)

    if angle>180:
        angle=360-angle

    return angle


image=cv2.imread("incorrect_squat.webp")

rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

mp_image=mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

result=detector.detect(mp_image)

landmarks=result.pose_landmarks[0]

def get_point(i):
    return [landmarks[i].x,landmarks[i].y]

shoulder=get_point(11)
hip=get_point(23)
knee=get_point(25)
ankle=get_point(27)

ear=get_point(7)

knee_angle=calculate_angle(hip,knee,ankle)
hip_angle=calculate_angle(shoulder,hip,knee)
back_angle=calculate_angle(ear,shoulder,hip)

features=pd.DataFrame([[knee_angle,hip_angle,back_angle]],
columns=["knee_angle","hip_angle","back_angle"])

prediction=model.predict(features)

print("Prediction:",prediction[0])