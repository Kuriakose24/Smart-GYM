import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.feedback_engine import FeedbackEngine


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
# Improved Thresholds
# -----------------------------

BOTTOM_THRESHOLD = 120
TOP_THRESHOLD = 135


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
# Angle Smoothing
# -----------------------------

def smooth(prev,current,alpha=0.7):

    return alpha*prev + (1-alpha)*current


# -----------------------------
# Feedback Engine
# -----------------------------

feedback_engine = FeedbackEngine()


# -----------------------------
# Camera
# -----------------------------

cap=cv2.VideoCapture(0)

timestamp=0

rep_count=0
correct_reps=0
incorrect_reps=0

stage="UP"
feedback="Stand in pushup position"

bottom_reached=False
exercise_started=False


# bottom posture storage
bottom_elbow=0
bottom_body=0
bottom_hip=0
bottom_back=0


# smoothing memory
prev_elbow=0
prev_body=0
prev_hip=0
prev_back=0


print("Press Q to exit")


# -----------------------------
# Main Loop
# -----------------------------

while cap.isOpened():

    ret,frame=cap.read()

    if not ret:
        break

    h,w,_=frame.shape
    timestamp+=1

    image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image=mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image
    )

    result=detector.detect_for_video(mp_image,timestamp)

    if result.pose_landmarks:

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


        # -----------------------------
        # Raw Angles
        # -----------------------------

        raw_elbow=calculate_angle(shoulder,elbow,wrist)
        raw_body=calculate_angle(shoulder,hip,ankle)
        raw_hip=calculate_angle(shoulder,hip,knee)
        raw_back=calculate_angle(ear,shoulder,hip)


        # -----------------------------
        # Smooth Angles
        # -----------------------------

        elbow_angle=smooth(prev_elbow,raw_elbow)
        body_angle=smooth(prev_body,raw_body)
        hip_angle=smooth(prev_hip,raw_hip)
        back_angle=smooth(prev_back,raw_back)

        prev_elbow=elbow_angle
        prev_body=body_angle
        prev_hip=hip_angle
        prev_back=back_angle


        # -----------------------------
        # Orientation Check
        # -----------------------------

        horizontal_body = abs(shoulder[1] - hip[1]) < 0.20


        # -----------------------------
        # Detect Pushup Start
        # -----------------------------

        if horizontal_body and elbow_angle > 160 and not exercise_started:

            exercise_started=True
            feedback="Pushup position detected"


        # -----------------------------
        # Rep Detection
        # -----------------------------

        if horizontal_body and exercise_started:

            if elbow_angle < BOTTOM_THRESHOLD + 5:

                stage="DOWN"
                bottom_reached=True

                bottom_elbow = elbow_angle
                bottom_body = body_angle
                bottom_hip = hip_angle
                bottom_back = back_angle


            if elbow_angle > TOP_THRESHOLD and stage=="DOWN" and bottom_reached:

                stage="UP"
                bottom_reached=False

                rep_count+=1


                # -----------------------------
                # ML Prediction
                # -----------------------------

                features=pd.DataFrame(
                    [[bottom_elbow,bottom_back,bottom_hip,bottom_body]],
                    columns=[
                        "elbow_angle",
                        "back_angle",
                        "hip_angle",
                        "knee_angle"
                    ]
                )

                prediction=model.predict(features)[0]


                # -----------------------------
                # Rule Override
                # -----------------------------

                rule_violation=False

                if bottom_hip < 150:
                    prediction="incorrect"
                    feedback="Hips too high"
                    rule_violation=True

                elif bottom_hip > 200:
                    prediction="incorrect"
                    feedback="Hips sagging"
                    rule_violation=True

                elif bottom_elbow > 125:
                    prediction="incorrect"
                    feedback="Go lower"
                    rule_violation=True


                # -----------------------------
                # Rep Score
                # -----------------------------

                score=feedback_engine.calculate_rep_score(
                    bottom_elbow,
                    bottom_body,
                    bottom_hip,
                    bottom_back
                )

                feedback_engine.add_rep_score(score)


                # -----------------------------
                # Final Feedback
                # -----------------------------

                if prediction=="correct":

                    correct_reps+=1
                    feedback="Perfect Rep"

                else:

                    incorrect_reps+=1

                    if not rule_violation:
                        feedback=feedback_engine.generate_feedback(
                            bottom_elbow,
                            bottom_body,
                            bottom_hip,
                            bottom_back
                        )


        else:

            if not exercise_started:
                feedback="Get into pushup position"

            stage="UP"


        # -----------------------------
        # Draw Landmarks
        # -----------------------------

        for lm in landmarks:

            x=int(lm.x*w)
            y=int(lm.y*h)

            cv2.circle(frame,(x,y),3,(0,255,0),-1)


        # -----------------------------
        # Debug Angles
        # -----------------------------

        cv2.putText(frame,f"Hip:{int(hip_angle)}",(30,190),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.putText(frame,f"Elbow:{int(elbow_angle)}",(30,220),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)


        # -----------------------------
        # Display
        # -----------------------------

        cv2.putText(frame,f"Reps: {rep_count}",(30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

        cv2.putText(frame,f"Correct: {correct_reps}",(30,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.putText(frame,f"Incorrect: {incorrect_reps}",(30,110),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        cv2.putText(frame,f"Feedback: {feedback}",(30,150),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


    cv2.imshow("Smart Gym Pushup Engine",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break


summary=feedback_engine.workout_summary()

print("\nWorkout Summary")
print(summary)

if summary["trainer_alert"]:
    print("Trainer Alert: Posture improvement needed")
else:
    print("Perfect Workout")


cap.release()
cv2.destroyAllWindows()