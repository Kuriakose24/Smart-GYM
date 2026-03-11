import sys, time
sys.path.append('.')
from camera.video_stream import VideoStream
from tracking.person_tracker import PersonTracker
from pose.pose_estimator import PoseEstimator

cam       = VideoStream(source=1)
tracker   = PersonTracker()
estimator = PoseEstimator()
cam.start()

print("Just stand still normally for 15 seconds")
print("Elbow | Knee | Hip | Horizontal\n")

for i in range(150):
    ret, frame = cam.read()
    tracked = tracker.update(frame)
    for p in tracked:
        angles = estimator.extract(p["keypoints"])
        if angles:
            e  = angles.get("elbow", 0)
            k  = angles.get("knee",  0)
            h  = angles.get("hip",   0)
            hz = angles.get("horizontal")
            print(f"Elbow:{e:.0f}  Knee:{k:.0f}  Hip:{h:.0f}  Horiz:{hz}")
    time.sleep(0.1)

cam.stop()