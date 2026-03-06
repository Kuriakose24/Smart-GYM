import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

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


def detect_stage(elbow):

    if elbow >= 150:
        return "top"

    elif 80 <= elbow <= 120:
        return "bottom"

    else:
        return "moving"


image_folder="dataset/image_dataset"

data=[]

for img in os.listdir(image_folder):

    path=os.path.join(image_folder,img)

    frame=cv2.imread(path)

    if frame is None:
        continue

    image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image=mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    result=detector.detect(mp_image)

    if not result.pose_landmarks:
        continue

    landmarks=result.pose_landmarks[0]

    def get_point(i):
        return [landmarks[i].x,landmarks[i].y]

    shoulder=get_point(11)
    elbow=get_point(13)
    wrist=get_point(15)

    hip=get_point(23)
    knee=get_point(25)
    ankle=get_point(27)

    ear=get_point(7)


    elbow_angle=calculate_angle(shoulder,elbow,wrist)
    back_angle=calculate_angle(ear,shoulder,hip)
    hip_angle=calculate_angle(shoulder,hip,knee)
    knee_angle=calculate_angle(hip,knee,ankle)


    stage=detect_stage(elbow_angle)


    # ignore moving frames
    if stage=="moving":
        continue


    good_back = back_angle >= 135
    good_hip = hip_angle >= 135


    if good_back and good_hip:
        label="correct"
    else:
        label="incorrect"


    data.append([
        elbow_angle,
        back_angle,
        hip_angle,
        knee_angle,
        stage,
        label
    ])


df=pd.DataFrame(data,columns=[
    "elbow_angle",
    "back_angle",
    "hip_angle",
    "knee_angle",
    "stage",
    "label"
])

df.to_csv("pushup_image_dataset.csv",index=False)

print("Dataset created")
print("Samples:",len(df))