import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Load ML Model
# -----------------------------

model = joblib.load("models/pushup_final_model.pkl")


# -----------------------------
# Pose Detector
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path="models/pose_landmarker.task"
)

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
# Rep Score Calculation
# -----------------------------

def calculate_rep_score(elbow, body, hip, back):

    score = 0

    # body alignment (priority)
    if 160 <= body <= 200:
        score += 40
    elif 140 <= body < 160:
        score += 25
    else:
        score += 10

    # hip alignment
    if hip > 160:
        score += 25
    elif hip > 140:
        score += 15
    else:
        score += 5

    # back
    if back > 150:
        score += 20
    elif back > 130:
        score += 10
    else:
        score += 5

    # depth
    if elbow < 90:
        score += 15
    elif elbow < 110:
        score += 10
    else:
        score += 5

    return score


# -----------------------------
# Feedback Priority Engine
# -----------------------------

def generate_feedback(elbow, body, hip, back):

    if body < 140:
        return "Hips sagging"

    elif body > 210:
        return "Hips too high"

    elif elbow > 100:
        return "Lower your body"

    elif back < 120:
        return "Keep back straight"

    else:
        return "Good form"


# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

timestamp = 0

rep_count = 0
correct_reps = 0
incorrect_reps = 0

stage = "UP"

feedback = "Stand in pushup position"

rep_scores = []

print("Press Q to exit")


# -----------------------------
# Main Loop
# -----------------------------

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
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
        ankle = get_point(27)

        ear = get_point(7)


        # -----------------------------
        # Angle Calculations
        # -----------------------------

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        body_angle = calculate_angle(shoulder, hip, ankle)
        hip_angle = calculate_angle(shoulder, hip, knee)
        back_angle = calculate_angle(ear, shoulder, hip)


        # -----------------------------
        # Orientation Filter
        # -----------------------------

        horizontal_body = abs(shoulder[1] - hip[1]) < 0.20


        if horizontal_body:

            # DOWN
            if elbow_angle < 95:
                stage = "DOWN"

            # REP COMPLETE
            if elbow_angle > 150 and stage == "DOWN":

                stage = "UP"

                rep_count += 1


                # -----------------------------
                # Rep Score
                # -----------------------------

                score = calculate_rep_score(
                    elbow_angle,
                    body_angle,
                    hip_angle,
                    back_angle
                )

                rep_scores.append(score)


                # -----------------------------
                # ML Prediction
                # -----------------------------

                features = pd.DataFrame(
                    [[elbow_angle, back_angle, hip_angle, body_angle]],
                    columns=[
                        "elbow_angle",
                        "back_angle",
                        "hip_angle",
                        "knee_angle"
                    ]
                )

                prediction = model.predict(features)[0]


                # -----------------------------
                # Feedback
                # -----------------------------

                if prediction == "correct":

                    correct_reps += 1
                    feedback = "Perfect Rep"

                else:

                    incorrect_reps += 1
                    feedback = generate_feedback(
                        elbow_angle,
                        body_angle,
                        hip_angle,
                        back_angle
                    )


        else:

            feedback = "Stand in pushup position"
            stage = "UP"


        # -----------------------------
        # Draw Landmarks
        # -----------------------------

        for lm in landmarks:

            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(frame, (x,y), 3, (0,255,0), -1)


        # -----------------------------
        # Display
        # -----------------------------

        cv2.putText(frame,f"Reps: {rep_count}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,0,0),2)

        cv2.putText(frame,f"Correct: {correct_reps}",
                    (30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

        cv2.putText(frame,f"Incorrect: {incorrect_reps}",
                    (30,110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,0,255),2)

        cv2.putText(frame,f"Feedback: {feedback}",
                    (30,150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(255,255,255),2)


    cv2.imshow("Smart Gym Pushup Engine",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# -----------------------------
# Workout Summary
# -----------------------------

if len(rep_scores) > 0:

    avg_score = sum(rep_scores) / len(rep_scores)

    print("\nWorkout Summary")
    print("Total Reps:",rep_count)
    print("Average Score:",round(avg_score,2))

    if avg_score < 75:
        print("⚠ Trainer Alert: Workout posture needs improvement")
    else:
        print("✔ Perfect Workout")


cap.release()
cv2.destroyAllWindows()