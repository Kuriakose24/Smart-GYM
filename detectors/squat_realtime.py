import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------
# Load Squat ML Model
# --------------------------

model = joblib.load("models/squat_final_model.pkl")


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
# Angle Smoothing
# --------------------------

def smooth(prev,current,alpha=0.7):
    return alpha*prev + (1-alpha)*current


prev_knee=0


# --------------------------
# Bottom posture tracking
# --------------------------

bottom_knee=180
bottom_hip=0
bottom_back=0


# --------------------------
# Rep logic variables
# --------------------------

exercise_started=False
down_frames=0

stage="top"

rep_count=0
correct_reps=0
incorrect_reps=0

feedback="Stand straight"


# --------------------------
# Camera
# --------------------------

cap=cv2.VideoCapture(0)

timestamp=0

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
        hip=get_point(23)
        knee=get_point(25)
        ankle=get_point(27)

        ear=get_point(7)


        # --------------------------
        # Calculate Angles
        # --------------------------

        raw_knee=calculate_angle(hip,knee,ankle)
        hip_angle=calculate_angle(shoulder,hip,knee)
        back_angle=calculate_angle(ear,shoulder,hip)


        knee_angle=smooth(prev_knee,raw_knee)
        prev_knee=knee_angle


        # --------------------------
        # Detect start position
        # --------------------------

        if knee_angle > 165 and not exercise_started:

            exercise_started=True
            feedback="Squat position detected"


        # --------------------------
        # Squat down detection
        # --------------------------

        if exercise_started and knee_angle < 130:

            down_frames += 1
            stage="down"

            if knee_angle < bottom_knee:

                bottom_knee=knee_angle
                bottom_hip=hip_angle
                bottom_back=back_angle

        else:

            down_frames=0


        # --------------------------
        # Stand up detection
        # --------------------------

        if exercise_started and knee_angle > 165 and stage=="down" and down_frames>4:

            rep_count+=1
            stage="top"


            features=pd.DataFrame(
                [[bottom_knee,bottom_hip,bottom_back]],
                columns=[
                    "knee_angle",
                    "hip_angle",
                    "back_angle"
                ]
            )

            prediction=model.predict(features)[0]


            if prediction=="correct":

                correct_reps+=1
                feedback="Good Squat"

            else:

                incorrect_reps+=1
                feedback="Fix posture"


            bottom_knee=180


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
                    0.7,(255,255,255),2)


    cv2.imshow("Smart Gym Squat Detection",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()