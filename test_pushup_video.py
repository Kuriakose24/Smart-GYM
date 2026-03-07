import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib
from collections import Counter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------
# Load ML model
# --------------------------

model = joblib.load("pushup_final_model.pkl")


# --------------------------
# Pose detector
# --------------------------

base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# --------------------------
# Angle calculation
# --------------------------

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle

    return angle


# --------------------------
# Video input
# --------------------------

cap = cv2.VideoCapture("test_pushup.mp4")

timestamp = 0
stage = "top"

rep_predictions = []
rep_count = 0
rep_result = "Analyzing"

print("Press Q to exit")


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Video finished or frame not read")
        break

    h, w, _ = frame.shape
    timestamp += 1

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


        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        body_angle = calculate_angle(shoulder, hip, knee)
        hip_angle = calculate_angle(shoulder, hip, knee)


        # --------------------------
        # Stage detection
        # --------------------------

        if elbow_angle < 95:
            stage = "down"

        if elbow_angle > 150 and stage == "down":

            # REP COMPLETED

            if len(rep_predictions) > 5:

                rep_count += 1

                majority = Counter(rep_predictions).most_common(1)[0][0]

                rep_result = majority

            rep_predictions = []
            stage = "top"


        # --------------------------
        # Frame prediction
        # --------------------------

        if stage == "down" and elbow_angle < 90:

            if body_angle < 135:
                rule_prediction = "incorrect"

            elif hip_angle < 130:
                rule_prediction = "incorrect"

            else:
                rule_prediction = "correct"


            features = pd.DataFrame(
                [[elbow_angle, 0, hip_angle, body_angle]],
                columns=["elbow_angle", "back_angle", "hip_angle", "knee_angle"]
            )

            ml_prediction = model.predict(features)[0]


            if rule_prediction == "correct" and ml_prediction == "correct":
                prediction = "correct"
            else:
                prediction = "incorrect"


            rep_predictions.append(prediction)


        # --------------------------
        # Draw landmarks
        # --------------------------

        for lm in landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)


        # --------------------------
        # Display information
        # --------------------------

        cv2.putText(frame, f"Reps: {rep_count}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2)

        cv2.putText(frame, f"Last Rep: {rep_result}",
                    (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        cv2.putText(frame, f"Elbow: {int(elbow_angle)}",
                    (30, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Body: {int(body_angle)}",
                    (30, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Hip: {int(hip_angle)}",
                    (30, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)


    cv2.imshow("Smart Gym Pushup Detection", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()