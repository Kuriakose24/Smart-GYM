import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib
from collections import deque

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
# Angle smoothing
# --------------------------

def smooth(prev,current,alpha=0.7):
    return alpha*prev + (1-alpha)*current


# --------------------------
# Video input
# --------------------------

cap=cv2.VideoCapture("test_pushup.mp4")

timestamp=0

prev_elbow=0
prev_body=0

prediction_buffer=deque(maxlen=8)

stage="top"


print("Press Q to exit")


# --------------------------
# Main loop
# --------------------------

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
        # Calculate angles
        # --------------------------

        raw_elbow=calculate_angle(shoulder,elbow,wrist)

        raw_body=calculate_angle(shoulder,hip,ankle)

        hip_angle=calculate_angle(shoulder,hip,knee)

        back_angle=calculate_angle(ear,shoulder,hip)


        # --------------------------
        # Smooth angles
        # --------------------------

        elbow_angle=smooth(prev_elbow,raw_elbow)

        body_angle=smooth(prev_body,raw_body)

        prev_elbow=elbow_angle
        prev_body=body_angle


        # --------------------------
        # Stage detection
        # --------------------------

        if elbow_angle < 110:
            stage="down"

        if elbow_angle > 150:
            stage="top"


        posture="analyzing"


        # --------------------------
        # Only analyze bottom stage
        # --------------------------

        if stage=="down":

            # biomechanical rule check

            if body_angle < 150:
                prediction="incorrect"

            elif hip_angle < 140:
                prediction="incorrect"

            else:

                features=pd.DataFrame(
                    [[elbow_angle,back_angle,hip_angle,body_angle]],
                    columns=["elbow_angle","back_angle","hip_angle","knee_angle"]
                )

                prediction=model.predict(features)[0]


            prediction_buffer.append(prediction)


            # frame voting

            posture=max(set(prediction_buffer),
                        key=prediction_buffer.count)


        # --------------------------
        # Draw landmarks
        # --------------------------

        for lm in landmarks:

            x=int(lm.x*w)
            y=int(lm.y*h)

            cv2.circle(frame,(x,y),3,(0,255,0),-1)


        # --------------------------
        # Display info
        # --------------------------

        cv2.putText(frame,f"Stage: {stage}",
                    (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,0,0),2)

        cv2.putText(frame,f"Posture: {posture}",
                    (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,0,255),2)


        cv2.putText(frame,f"Elbow:{int(elbow_angle)}",
                    (30,160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,255),2)

        cv2.putText(frame,f"Body:{int(body_angle)}",
                    (30,190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,(255,255,255),2)


    cv2.imshow("Stable Pushup Analysis",frame)


    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()