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

model = joblib.load("models/squat_final_model.pkl")


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
# Rep Thresholds
# -----------------------------

DOWN_THRESHOLD = 130
UP_THRESHOLD = 160


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

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

timestamp=0

rep_count=0
correct_reps=0
incorrect_reps=0

stage="UP"
feedback="Stand straight"

exercise_started=False


# deepest squat tracking
deepest_knee=180
deepest_hip=0
deepest_back=0


# smoothing memory
prev_knee=0
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
    timestamp+=33

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
        hip=get_point(23)
        knee=get_point(25)
        ankle=get_point(27)

        ear=get_point(7)


        # -----------------------------
        # Raw Angles
        # -----------------------------

        raw_knee=calculate_angle(hip,knee,ankle)
        raw_hip=calculate_angle(shoulder,hip,knee)
        raw_back=calculate_angle(ear,shoulder,hip)


        # -----------------------------
        # Smooth Angles
        # -----------------------------

        knee_angle=smooth(prev_knee,raw_knee)
        hip_angle=smooth(prev_hip,raw_hip)
        back_angle=smooth(prev_back,raw_back)

        prev_knee=knee_angle
        prev_hip=hip_angle
        prev_back=back_angle


        # -----------------------------
        # Detect Squat Start
        # -----------------------------

        if knee_angle > 165 and not exercise_started:

            exercise_started=True
            feedback="Squat position detected"


        # -----------------------------
        # Rep Detection
        # -----------------------------

        if exercise_started:

            if knee_angle < DOWN_THRESHOLD:

                stage="DOWN"

                if knee_angle < deepest_knee:

                    deepest_knee=knee_angle
                    deepest_hip=hip_angle
                    deepest_back=back_angle


            if knee_angle > UP_THRESHOLD and stage=="DOWN":

                stage="UP"
                rep_count+=1


                # -----------------------------
                # ML Prediction
                # -----------------------------

                features=pd.DataFrame(
                    [[deepest_knee,deepest_hip,deepest_back]],
                    columns=[
                        "knee_angle",
                        "hip_angle",
                        "back_angle"
                    ]
                )

                prediction=model.predict(features)[0]


                # -----------------------------
                # Rule Override
                # -----------------------------

                rule_violation=False

                if deepest_knee > 120:
                    prediction="incorrect"
                    feedback="Go deeper"
                    rule_violation=True

                elif deepest_back < 130:
                    prediction="incorrect"
                    feedback="Keep chest up"
                    rule_violation=True

                elif deepest_hip < 110:
                    prediction="incorrect"
                    feedback="Do not lean forward"
                    rule_violation=True


                # -----------------------------
                # Rep Score
                # -----------------------------

                score=feedback_engine.calculate_rep_score(
                    deepest_knee,
                    deepest_back,
                    deepest_hip,
                    0
                )

                feedback_engine.add_rep_score(score)


                # -----------------------------
                # Final Feedback
                # -----------------------------

                if prediction=="correct":

                    correct_reps+=1
                    feedback="Good Squat"

                else:

                    incorrect_reps+=1

                    if not rule_violation:
                        feedback=feedback_engine.generate_feedback(
                            deepest_knee,
                            deepest_back,
                            deepest_hip,
                            0
                        )


                # reset deepest values
                deepest_knee=180


        else:

            feedback="Stand straight"


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

        cv2.putText(frame,f"Knee:{int(knee_angle)}",(30,200),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        cv2.putText(frame,f"Hip:{int(hip_angle)}",(30,230),
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


    cv2.imshow("Smart Gym Squat Engine",frame)

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