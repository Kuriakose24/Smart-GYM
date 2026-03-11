import sys, time
sys.path.append('.')
from camera.video_stream import VideoStream
from tracking.person_tracker import PersonTracker
from pose.pose_estimator import PoseEstimator
import config

cam       = VideoStream(source=config.CAMERA_INDEX)
tracker   = PersonTracker()
estimator = PoseEstimator()
cam.start()

print("Stand straight for 5 sec, then slowly squat down, then stand back up")
print("Knee | Hip | Elbow\n")

for i in range(300):
    ret, frame = cam.read()
    if not ret:
        continue
    tracked = tracker.update(frame)
    for p in tracked:
        angles = estimator.extract(p["keypoints"])
        if angles and "knee" in angles:
            knee  = angles["knee"]
            hip   = angles.get("hip", 0)
            elbow = angles.get("elbow", 0)
            print(f"Knee:{knee:.0f}  Hip:{hip:.0f}  Elbow:{elbow:.0f}")
    time.sleep(0.1)

cam.stop()