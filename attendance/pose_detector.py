"""
pose_detector.py
----------------
Uses YOLOv8-pose to detect persons AND their 17 body keypoints
in a single pass — no separate MediaPipe needed.

YOLOv8-pose keypoint indices:
    0  = nose
    5  = left shoulder     6  = right shoulder
    7  = left elbow        8  = right elbow
    9  = left wrist        10 = right wrist
    11 = left hip          12 = right hip
    13 = left knee         14 = right knee
    15 = left ankle        16 = right ankle

Returns per person:
    box       : (x1, y1, x2, y2)
    keypoints : dict of named points → (x, y) in pixel coords
    confidence: float
"""

import cv2
import numpy as np
from ultralytics import YOLO


# Keypoint index mapping — human readable names
KP = {
    "nose":           0,
    "left_shoulder":  5,
    "right_shoulder": 6,
    "left_elbow":     7,
    "right_elbow":    8,
    "left_wrist":     9,
    "right_wrist":    10,
    "left_hip":       11,
    "right_hip":      12,
    "left_knee":      13,
    "right_knee":     14,
    "left_ankle":     15,
    "right_ankle":    16,
    "left_ear":       3,
    "right_ear":      4,
}


class PoseDetector:
    def __init__(self, model_path="yolov8n-pose.pt", confidence=0.5):
        """
        model_path : YOLOv8 pose model (auto-downloads on first run ~6MB)
        confidence : minimum detection confidence
        """
        self.confidence = confidence
        print(f"[PoseDetector] Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("[PoseDetector] Model loaded.")

    def detect(self, frame):
        """
        Run pose detection on frame.

        Returns list of dicts:
        [
            {
                "box":        (x1, y1, x2, y2),
                "confidence": 0.91,
                "keypoints":  {
                    "left_shoulder": (x, y),
                    "left_elbow":    (x, y),
                    ...
                }
            },
            ...
        ]
        Only persons with confidence >= threshold returned.
        Keypoints with low visibility are set to None.
        """
        results = self.model(frame, verbose=False)
        persons = []

        for r in results:
            if r.keypoints is None:
                continue

            boxes = r.boxes
            kps   = r.keypoints.xy.cpu().numpy()    # shape: (N, 17, 2)
            confs = r.keypoints.conf                 # shape: (N, 17) or None

            for i in range(len(boxes)):
                conf = float(boxes[i].conf[0])
                if conf < self.confidence:
                    continue

                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])

                # Build named keypoint dict
                kp_dict = {}
                for name, idx in KP.items():
                    x, y = kps[i][idx]
                    # Check visibility if available
                    if confs is not None:
                        vis = float(confs[i][idx])
                        kp_dict[name] = (float(x), float(y)) if vis > 0.3 else None
                    else:
                        kp_dict[name] = (float(x), float(y))

                persons.append({
                    "box":        (x1, y1, x2, y2),
                    "confidence": conf,
                    "keypoints":  kp_dict
                })

        return persons

    def draw_skeleton(self, frame, persons):
        """Draw keypoints and skeleton lines on frame."""
        # Skeleton connections
        connections = [
            ("left_shoulder",  "right_shoulder"),
            ("left_shoulder",  "left_elbow"),
            ("left_elbow",     "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow",    "right_wrist"),
            ("left_shoulder",  "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip",       "right_hip"),
            ("left_hip",       "left_knee"),
            ("left_knee",      "left_ankle"),
            ("right_hip",      "right_knee"),
            ("right_knee",     "right_ankle"),
        ]

        for person in persons:
            kp = person["keypoints"]

            # Draw connections
            for a, b in connections:
                if kp.get(a) and kp.get(b):
                    cv2.line(frame,
                             (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])),
                             (0, 255, 0), 2)

            # Draw keypoints
            for name, pt in kp.items():
                if pt:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 200, 255), -1)

        return frame