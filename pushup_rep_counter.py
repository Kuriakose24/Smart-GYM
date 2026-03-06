import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------------------
# Load pose detector
# -------------------------------

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# -------------------------------
# Angle calculation
# -------------------------------

def calculate_angle(a,b,c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle


# -------------------------------
# Rep variables
# -------------------------------

rep_count = 0
stage = "top"

min_elbow = 180
feedback = ""


# -------------------------------
# Video input
# -------------------------------

cap = cv2.VideoCapture("pushup_test_video.mp4")
# webcam option
# cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open video")
    exit()


# -------------------------------
# Main loop
# -------------------------------

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    result = detector.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        def get_point(i):
            return [landmarks[i].x, landmarks[i].y]

        shoulder = get_point(11)
        elbow = get_point(13)
        wrist = get_point(15)

        hip = get_point(23)
        knee = get_point(25)
        ankle = get_point(27)

        ear = get_point(7)


        # -------------------------------
        # Calculate angles
        # -------------------------------

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        back_angle = calculate_angle(ear, shoulder, hip)

        hip_angle = calculate_angle(shoulder, hip, knee)

        knee_angle = calculate_angle(hip, knee, ankle)


        # -------------------------------
        # Track minimum elbow angle
        # -------------------------------

        if elbow_angle < min_elbow:
            min_elbow = elbow_angle


        # -------------------------------
        # Detect lowering phase
        # -------------------------------

        if elbow_angle < 110 and stage == "top":
            stage = "down"


        # -------------------------------
        # Detect bottom position
        # -------------------------------

        if elbow_angle < 95 and stage == "down":

            stage = "bottom"

            if hip_angle < 120:
                feedback = "Hip sagging"

            elif back_angle < 120:
                feedback = "Back bending"

            else:
                feedback = "Good push-up"


        # -------------------------------
        # Rep completion
        # -------------------------------

        if elbow_angle > 150 and stage == "bottom":

            rep_count += 1
            stage = "top"

            if min_elbow > 100:
                feedback = "Not deep enough"

            print("Rep", rep_count, ":", feedback)

            min_elbow = 180


        # -------------------------------
        # Draw results
        # -------------------------------

        cv2.putText(
            frame,
            f"Reps: {rep_count}",
            (30,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Stage: {stage}",
            (30,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2
        )

        cv2.putText(
            frame,
            f"Feedback: {feedback}",
            (30,160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )


        # -------------------------------
        # Show debug angles
        # -------------------------------

        cv2.putText(frame, f"Elbow: {int(elbow_angle)}",
                    (30,220), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,0), 2)

        cv2.putText(frame, f"Back: {int(back_angle)}",
                    (30,250), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,0), 2)

        cv2.putText(frame, f"Hip: {int(hip_angle)}",
                    (30,280), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255,255,0), 2)


    cv2.imshow("Smart Gym Push-up Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()