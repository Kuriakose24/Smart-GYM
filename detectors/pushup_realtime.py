import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------
# Load ML Model
# --------------------------

model = joblib.load("models/pushup_final_model.pkl")


# --------------------------
# Pose Detector
# --------------------------

base_options = python.BaseOptions(
    model_asset_path="models/pose_landmarker.task"
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.PoseLandmarker.create_from_options(options)


# --------------------------
# Angle Calculation
# --------------------------

def calculate_angle(a,b,c):

    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180/np.pi)

    if angle>180:
        angle=360-angle

    return angle


# --------------------------
# Camera
# --------------------------

cap=cv2.VideoCapture(0)

timestamp=0

rep_count=0
correct_reps=0
incorrect_reps=0

stage="UP"

feedback="Stand in pushup position"

print("Press Q to exit")


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


        # --------------------------
        # Angle Calculations
        # --------------------------

        elbow_angle=calculate_angle(shoulder,elbow,wrist)
        body_angle=calculate_angle(shoulder,hip,ankle)
        hip_angle=calculate_angle(shoulder,hip,knee)
        back_angle=calculate_angle(ear,shoulder,hip)


        # --------------------------
        # Orientation Filter
        # --------------------------

        horizontal_body = abs(shoulder[1] - hip[1]) < 0.20


        if horizontal_body:

            # DOWN position
            if elbow_angle < 95:

                stage="DOWN"


            # REP COMPLETE
            if elbow_angle > 150 and stage=="DOWN":

                stage="UP"
                rep_count+=1


                # --------------------------
                # ML Prediction
                # --------------------------

                features=pd.DataFrame(
                    [[elbow_angle,back_angle,hip_angle,body_angle]],
                    columns=["elbow_angle","back_angle","hip_angle","knee_angle"]
                )

                prediction=model.predict(features)[0]


                # --------------------------
                # Feedback Engine
                # --------------------------

                if prediction=="correct":

                    correct_reps+=1
                    feedback="Good form"

                else:

                    incorrect_reps+=1

                    if body_angle < 140:
                        feedback="Hips sagging"

                    elif body_angle > 210:
                        feedback="Hips too high"

                    elif elbow_angle > 100:
                        feedback="Lower your body"

                    elif back_angle < 120:
                        feedback="Keep back straight"

                    else:
                        feedback="Incorrect posture"

        else:

            feedback="Stand in pushup position"
            stage="UP"


        # --------------------------
        # Draw Landmarks
        # --------------------------

        for lm in landmarks:

            x=int(lm.x*w)
            y=int(lm.y*h)

            cv2.circle(frame,(x,y),3,(0,255,0),-1)


        # --------------------------
        # Display
        # --------------------------

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


    cv2.imshow("Smart Gym Pushup Feedback System",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()