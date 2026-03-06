import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Load posture model
# -----------------------------

model = joblib.load("pushup_posture_model.pkl")


# -----------------------------
# Load pose detector
# -----------------------------

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# -----------------------------
# Angle calculation
# -----------------------------

def calculate_angle(a,b,c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle


# -----------------------------
# Rep counter variables
# -----------------------------

rep_count = 0
stage = "up"

min_elbow = 180

hip_error_frames = 0
back_error_frames = 0

feedback = ""

# smoothing buffer
elbow_history = []


# -----------------------------
# Video input
# -----------------------------

cap = cv2.VideoCapture("dataset/videofiles/verified_data/verified_data/data_btc_10s/push-up/267bfdf2-42a2-4127-b9e1-683b58128a8a.mp4")
# webcam option:
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video")
    exit()


# -----------------------------
# Main loop
# -----------------------------

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

        if landmarks[11].visibility < 0.5:
            continue

        def get_point(i):
            return [landmarks[i].x, landmarks[i].y]

        shoulder = get_point(11)
        elbow = get_point(13)
        wrist = get_point(15)

        hip = get_point(23)
        knee = get_point(25)
        ankle = get_point(27)

        ear = get_point(7)


        # -----------------------------
        # Calculate angles
        # -----------------------------

        elbow_angle = calculate_angle(shoulder, elbow, wrist)

        hip_angle = calculate_angle(shoulder, hip, knee)

        back_angle = calculate_angle(ear, shoulder, hip)

        knee_angle = calculate_angle(hip, knee, ankle)


        # -----------------------------
        # Smooth elbow angle
        # -----------------------------

        elbow_history.append(elbow_angle)

        if len(elbow_history) > 5:
            elbow_history.pop(0)

        elbow_angle = sum(elbow_history) / len(elbow_history)


        # -----------------------------
        # Track lowest elbow
        # -----------------------------

        if elbow_angle < min_elbow:
            min_elbow = elbow_angle


        # -----------------------------
        # Detect down phase
        # -----------------------------

        if elbow_angle < 90:
            stage = "down"


        # -----------------------------
        # Human-friendly tolerance
        # -----------------------------

        if hip_angle < 120:
            hip_error_frames += 1

        if back_angle < 120:
            back_error_frames += 1


        # -----------------------------
        # Rep completion
        # -----------------------------

        if elbow_angle > 160 and stage == "down":

            rep_count += 1
            stage = "up"

            if min_elbow > 100:
                feedback = "Not deep enough"

            elif hip_error_frames > 20:
                feedback = "Hip sagging"

            elif back_error_frames > 20:
                feedback = "Back bending"

            else:
                feedback = "Good push-up"

            print("Rep", rep_count, ":", feedback)

            # reset
            min_elbow = 180
            hip_error_frames = 0
            back_error_frames = 0


        # -----------------------------
        # Model prediction
        # -----------------------------

        sample = pd.DataFrame(
            [[elbow_angle, back_angle, hip_angle, knee_angle]],
            columns=[
                "elbow_angle",
                "back_angle",
                "hip_angle",
                "knee_angle"
            ]
        )

        prediction = model.predict(sample)[0]


        # -----------------------------
        # Draw pose
        # -----------------------------

        h, w, _ = frame.shape

        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)


        # -----------------------------
        # Display info
        # -----------------------------

        cv2.putText(frame, f"Reps: {rep_count}",
                    (30,80), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.putText(frame, f"Stage: {stage}",
                    (30,120), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,0,0), 2)

        cv2.putText(frame, f"Posture: {prediction}",
                    (30,160), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,255), 2)

        cv2.putText(frame, f"Feedback: {feedback}",
                    (30,200), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)


        # -----------------------------
        # Show angles (debugging)
        # -----------------------------

        cv2.putText(frame, f"Back:{int(back_angle)}",
                    (30,240), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,0), 2)

        cv2.putText(frame, f"Hip:{int(hip_angle)}",
                    (30,270), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,0), 2)

        cv2.putText(frame, f"Elbow:{int(elbow_angle)}",
                    (30,300), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255,255,0), 2)


    cv2.imshow("Smart Gym Pushup Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()