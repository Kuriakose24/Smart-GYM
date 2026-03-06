import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_pose = mp.tasks.vision.PoseLandmarker
BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

pose = mp_pose.create_from_options(options)

dataset = []

video_folder = "dataset/videofiles/verified_data/verified_data/data_btc_10s/push-up"

def calculate_angle(a,b,c):

    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)

    if angle>180:
        angle=360-angle

    return angle


for video in os.listdir(video_folder):

    path=os.path.join(video_folder,video)

    cap=cv2.VideoCapture(path)

    frame_id=0

    while True:

        ret,frame=cap.read()

        if not ret:
            break

        frame_id+=1

        if frame_id % 10 != 0:
            continue

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        
        results = pose.detect(image)

        if results.pose_landmarks:

            lm = results.pose_landmarks[0]

            shoulder=[lm[11].x,lm[11].y]
            elbow=[lm[13].x,lm[13].y]
            wrist=[lm[15].x,lm[15].y]

            hip=[lm[23].x,lm[23].y]
            knee=[lm[25].x,lm[25].y]
            ankle=[lm[27].x,lm[27].y]

            elbow_angle=calculate_angle(shoulder,elbow,wrist)
            back_angle=calculate_angle(shoulder,hip,ankle)
            hip_angle=calculate_angle(shoulder,hip,knee)
            knee_angle=calculate_angle(hip,knee,ankle)

            dataset.append([
                elbow_angle,
                back_angle,
                hip_angle,
                knee_angle,
                "pushup"
            ])

    cap.release()

df=pd.DataFrame(dataset,
columns=[
"elbow_angle",
"back_angle",
"hip_angle",
"knee_angle",
"exercise"
])

df.to_csv("pushup_video_dataset.csv",index=False)

print("Dataset created successfully")