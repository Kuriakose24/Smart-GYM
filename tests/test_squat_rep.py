import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------
# Pose detector
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
# Rep Variables
# --------------------------

rep_count=0
stage="UP"

deepest_knee=180


# --------------------------
# Video
# --------------------------

cap=cv2.VideoCapture("squat_test.mp4")

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

        lm=result.pose_landmarks[0]

        def get_point(i):
            return [lm[i].x,lm[i].y]


        shoulder=get_point(11)
        hip=get_point(23)
        knee=get_point(25)
        ankle=get_point(27)


        # calculate knee angle
        knee_angle=calculate_angle(hip,knee,ankle)


        # --------------------------
        # Squat Down
        # --------------------------

        if knee_angle < 130:

            stage="DOWN"

            if knee_angle < deepest_knee:
                deepest_knee=knee_angle


        # --------------------------
        # Stand Up (rep complete)
        # --------------------------

        if knee_angle > 160 and stage=="DOWN":

            rep_count+=1
            stage="UP"

            print("Rep:",rep_count," Bottom Knee:",deepest_knee)

            deepest_knee=180


        # draw landmarks
        for p in lm:

            x=int(p.x*w)
            y=int(p.y*h)

            cv2.circle(frame,(x,y),3,(0,255,0),-1)


        # display rep
        cv2.putText(frame,f"Reps: {rep_count}",
                    (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(255,0,0),2)


    cv2.imshow("Squat Rep Test",frame)

    if cv2.waitKey(1)&0xFF==ord('q'):
        break


cap.release()
cv2.destroyAllWindows()