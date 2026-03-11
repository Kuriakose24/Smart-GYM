"""
detection/person_detector.py
-----------------------------
Uses YOLOv8-pose to detect ALL persons in a frame AND their
17 body keypoints in a single pass.

This replaces BOTH detector.py and pose_detector.py from your old code.
One model, one call, two outputs: bounding boxes + keypoints.

YOLOv8-pose keypoint indices:
    0=nose   1=left_eye   2=right_eye   3=left_ear    4=right_ear
    5=left_shoulder       6=right_shoulder
    7=left_elbow          8=right_elbow
    9=left_wrist          10=right_wrist
    11=left_hip           12=right_hip
    13=left_knee          14=right_knee
    15=left_ankle         16=right_ankle

Each detected person returns:
    {
        "box":        (x1, y1, x2, y2),
        "confidence": 0.91,
        "keypoints":  {
            "left_shoulder": (x, y),   ← pixel coords in FULL frame
            "left_elbow":    (x, y),
            ...                         ← None if keypoint not visible
        }
    }
"""

import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Keypoint index → human readable name ──────────────────────────────────────
KP_NAMES = {
    "nose":            0,
    "left_eye":        1,
    "right_eye":       2,
    "left_ear":        3,
    "right_ear":       4,
    "left_shoulder":   5,
    "right_shoulder":  6,
    "left_elbow":      7,
    "right_elbow":     8,
    "left_wrist":      9,
    "right_wrist":     10,
    "left_hip":        11,
    "right_hip":       12,
    "left_knee":       13,
    "right_knee":      14,
    "left_ankle":      15,
    "right_ankle":     16,
}

# ── Skeleton connections for drawing ──────────────────────────────────────────
SKELETON = [
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


class PersonDetector:
    def __init__(self,
                 model_path=config.YOLO_MODEL,
                 confidence=config.YOLO_CONFIDENCE,
                 device=config.YOLO_DEVICE):

        self.confidence = confidence
        self.device     = device

        print(f"[Detector] Loading YOLOv8-pose model: {model_path}")
        print(f"[Detector] Device: {device}")
        self.model = YOLO(model_path)
        print("[Detector] ✅ Model loaded.")

    def detect(self, frame):
        """
        Run pose detection on a full frame.

        Returns list of person dicts — see module docstring for structure.
        Empty list if no persons found.
        """
        results = self.model(
            frame,
            verbose=False,
            device=self.device
        )

        persons = []

        for r in results:
            if r.keypoints is None or r.boxes is None:
                continue

            boxes  = r.boxes
            kp_xy  = r.keypoints.xy.cpu().numpy()       # shape: (N, 17, 2)
            kp_conf = r.keypoints.conf                   # shape: (N, 17) or None
            if kp_conf is not None:
                kp_conf = kp_conf.cpu().numpy()

            for i in range(len(boxes)):
                conf = float(boxes[i].conf[0])
                if conf < self.confidence:
                    continue

                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])

                # Build named keypoint dict
                kp_dict = {}
                for name, idx in KP_NAMES.items():
                    x, y = kp_xy[i][idx]
                    if kp_conf is not None:
                        vis = float(kp_conf[i][idx])
                        # Only include keypoint if visible enough
                        kp_dict[name] = (float(x), float(y)) if vis > 0.3 else None
                    else:
                        kp_dict[name] = (float(x), float(y))

                persons.append({
                    "box":        (x1, y1, x2, y2),
                    "confidence": conf,
                    "keypoints":  kp_dict
                })

        return persons

    def draw(self, frame, persons, identity_map=None):
        """
        Draw bounding boxes, skeleton, and labels on frame.

        identity_map : optional dict { person_index → name }
                       If provided, shows name instead of "Person"
        """
        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person["box"]
            kp = person["keypoints"]

            # Get name from identity map if available
            name  = identity_map.get(i, "Unknown") if identity_map else "Person"
            color = config.COLOR_BOX_KNOWN if name != "Unknown" else config.COLOR_BOX_UNKNOWN

            # Bounding box
            if config.SHOW_BOUNDING_BOX:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Name label with background
            if config.SHOW_IDENTITY:
                label = f"{name} {person['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.FONT_THICKNESS
                )
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, label,
                            (x1 + 3, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            config.FONT_SCALE, (0, 0, 0), config.FONT_THICKNESS)

            # Skeleton lines
            if config.SHOW_SKELETON:
                for a, b in SKELETON:
                    if kp.get(a) and kp.get(b):
                        cv2.line(frame,
                                 (int(kp[a][0]), int(kp[a][1])),
                                 (int(kp[b][0]), int(kp[b][1])),
                                 config.COLOR_SKELETON, 2)

                # Keypoint dots
                for pt_name, pt in kp.items():
                    if pt:
                        cv2.circle(frame,
                                   (int(pt[0]), int(pt[1])),
                                   4, (0, 200, 255), -1)

        return frame


# ── Test — run this file directly ─────────────────────────────────────────────
# python detection/person_detector.py
if __name__ == "__main__":
    import time
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.video_stream import VideoStream

    print("=" * 50)
    print("  Person Detector Test — press Q to quit")
    print("  Stand in front of camera to see skeleton")
    print("=" * 50)

    cam      = VideoStream()
    detector = PersonDetector()

    cam.start()

    fps_start   = time.time()
    fps_count   = 0
    fps_display = 0.0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # Detect all persons
        persons = detector.detect(frame)

        # Draw boxes + skeleton
        frame = detector.draw(frame, persons)

        # FPS
        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_start   = time.time()

        # HUD
        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons detected: {len(persons)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Print keypoints for first person (useful for debugging angles)
        if persons:
            kp = persons[0]["keypoints"]
            visible = [k for k, v in kp.items() if v is not None]
            cv2.putText(frame, f"Keypoints visible: {len(visible)}/17",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, "Press Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("SmartGym — Person Detector Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()

    print(f"\n✅ Detector test complete.")
    print(f"   If you saw a green skeleton drawn on your body — everything is working.")
    print(f"   If FPS < 15 — that's fine, tracking in the next step will smooth it out.")