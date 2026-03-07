import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------
# Pose model
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
# Angle function
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
stage="UP"

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
        ankle=get_point(27)


        # --------------------------
        # Angles
        # --------------------------

        elbow_angle=calculate_angle(shoulder,elbow,wrist)
        body_angle=calculate_angle(shoulder,hip,ankle)


        # --------------------------
        # Push-up orientation filter
        # --------------------------

        horizontal_body = abs(shoulder[1] - hip[1]) < 0.20


        if horizontal_body:

            # DOWN position
            if elbow_angle < 95:

                stage="DOWN"


            # UP position → REP
            if elbow_angle > 150 and stage=="DOWN":

                stage="UP"
                rep_count+=1


        else:

            stage="UP"


        # --------------------------
        # Draw landmarks
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


    cv2.imshow("Pushup Rep Counter",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()