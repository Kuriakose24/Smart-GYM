import cv2
import numpy as np
import pandas as pd
import os
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Load Pose Model
# -----------------------------

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# -----------------------------
# Angle Calculation
# -----------------------------

def calculate_angle(a,b,c):

    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)

    if angle>180:
        angle=360-angle

    return angle


# -----------------------------
# Stage Detection
# -----------------------------

def detect_stage(elbow):

    if elbow >= 150:
        return "top"

    elif 80 <= elbow <= 120:
        return "bottom"

    else:
        return "moving"


data=[]

dataset_path="dataset/vedio_dataset"

global_timestamp = 0


# -----------------------------
# Process Videos
# -----------------------------

for label in ["correct","incorrect"]:

    folder=os.path.join(dataset_path,label)

    for video_file in os.listdir(folder):

        path=os.path.join(folder,video_file)

        cap=cv2.VideoCapture(path)

        frame_id=0

        print("Processing:", video_file)

        while cap.isOpened():

            ret,frame=cap.read()

            if not ret:
                break

            frame_id+=1

            # sample every 5th frame
            if frame_id % 5 != 0:
                continue


            global_timestamp += 33


            image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            mp_image=mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=image
            )


            result=detector.detect_for_video(mp_image,global_timestamp)


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


            if stage=="moving":
                continue


            data.append([
                elbow_angle,
                back_angle,
                hip_angle,
                knee_angle,
                stage,
                label
            ])


        cap.release()


# -----------------------------
# Save Dataset
# -----------------------------

df=pd.DataFrame(data,columns=[
    "elbow_angle",
    "back_angle",
    "hip_angle",
    "knee_angle",
    "stage",
    "label"
])

df.to_csv("pushup_video_dataset.csv",index=False)


print("\nVideo dataset created")
print("Total samples:",len(df))