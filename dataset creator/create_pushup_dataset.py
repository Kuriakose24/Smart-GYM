import os
import numpy as np
import pandas as pd

label_folder = "dataset/exercise.v1i.yolov8/train/labels"

dataset = []

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle


for file in os.listdir(label_folder):

    path = os.path.join(label_folder, file)

    with open(path) as f:
        values = list(map(float, f.read().split()))

    keypoints = values[5:]

    points = []

    for i in range(0, len(keypoints), 3):

        x = keypoints[i]
        y = keypoints[i+1]

        points.append([x, y])


    # important joints
    left_shoulder = points[5]
    right_shoulder = points[6]

    left_elbow = points[7]
    right_elbow = points[8]

    left_wrist = points[9]
    right_wrist = points[10]

    left_hip = points[11]
    right_hip = points[12]

    left_knee = points[13]
    right_knee = points[14]

    left_ankle = points[15]
    right_ankle = points[16]


    # angles
    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

    shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)

    back_angle = calculate_angle(left_shoulder, left_hip, left_ankle)

    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)


    # posture rule (range based)

    if (
        back_angle > 150 and
        60 <= elbow_angle <= 180 and
        hip_angle > 150
    ):
        label = "correct"
    else:
        label = "incorrect"


    dataset.append([
        elbow_angle,
        shoulder_angle,
        back_angle,
        hip_angle,
        knee_angle,
        label
    ])


df = pd.DataFrame(dataset, columns=[
    "elbow_angle",
    "shoulder_angle",
    "back_angle",
    "hip_angle",
    "knee_angle",
    "label"
])

df.to_csv("pushup_posture_dataset.csv", index=False)

print("Push-up dataset created successfully")