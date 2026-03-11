"""
tracking/person_tracker.py  (Production Grade v2)
--------------------------------------------------
Upgraded from ByteTrack to BoT-SORT + ReID.

Why BoT-SORT is better for gym tracking:
    ByteTrack : tracks by position only
                person goes horizontal → YOLO loses them → new ID
    BoT-SORT  : tracks by position + appearance (ReID)
                person goes horizontal → appearance matches → SAME ID

What is ReID?
    Re-Identification = matching people by how they LOOK
    (clothing color, body shape, height)
    Even if YOLO loses the box for 10 frames, when the person
    reappears BoT-SORT matches their appearance and gives back
    the SAME track ID.

This fixes the core problem:
    BEFORE: pushup position → ID changes → rep counter resets
    AFTER:  pushup position → appearance matched → ID stays → reps continue

BoT-SORT is built into ultralytics — no new packages needed.
"""

import sys
import os
import numpy as np
import cv2
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# Skeleton connections for drawing
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

KP_NAMES = {
    "nose":           0,  "left_eye":       1,  "right_eye":      2,
    "left_ear":       3,  "right_ear":       4,
    "left_shoulder":  5,  "right_shoulder":  6,
    "left_elbow":     7,  "right_elbow":     8,
    "left_wrist":     9,  "right_wrist":     10,
    "left_hip":       11, "right_hip":       12,
    "left_knee":      13, "right_knee":      14,
    "left_ankle":     15, "right_ankle":     16,
}


class PersonTracker:
    """
    Production grade tracker using BoT-SORT + ReID via ultralytics.

    Key difference from v1:
        v1: supervision ByteTrack (position only)
        v2: ultralytics BoT-SORT (position + appearance ReID)

    Output per frame — list of dicts:
        {
            "track_id":   3,
            "box":        (x1, y1, x2, y2),
            "confidence": 0.91,
            "keypoints":  {"left_shoulder": (x,y), ...}
        }
    """

    def __init__(self,
                 model_path=config.YOLO_MODEL,
                 confidence=config.YOLO_CONFIDENCE,
                 device=config.YOLO_DEVICE):

        self.confidence = confidence
        self.device     = device

        print(f"[Tracker] Loading BoT-SORT tracker with {model_path}...")
        print(f"[Tracker] Device: {device}")

        self.model = YOLO(model_path)

        # Track history for smoothing box positions
        # { track_id: deque of last N boxes }
        self._box_history  = {}
        self._SMOOTH_N     = 3

        # ReID appearance cache
        # { track_id: last seen crop }
        self._appearance   = {}

        print("[Tracker] BoT-SORT + ReID ready.")

    def update(self, frame):
        """
        Run BoT-SORT tracking on a frame.
        Combines detection + tracking in ONE call (faster than v1).

        frame : BGR numpy array

        Returns list of tracked person dicts.
        """
        # Use ultralytics built-in BoT-SORT tracker
        results = self.model.track(
            frame,
            persist=True,           # CRITICAL: maintains track state between frames
            tracker="botsort.yaml", # BoT-SORT with ReID
            conf=self.confidence,
            classes=[0],            # only persons
            verbose=False,
            device=self.device,
        )

        persons = []

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue

            boxes    = r.boxes
            kp_xy    = r.keypoints.xy.cpu().numpy()
            kp_conf  = r.keypoints.conf
            if kp_conf is not None:
                kp_conf = kp_conf.cpu().numpy()

            # Get track IDs (None if tracking failed this frame)
            if boxes.id is None:
                continue

            track_ids = boxes.id.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                conf     = float(boxes[i].conf[0])
                track_id = int(track_ids[i])
                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])

                # Smooth box position
                box = self._smooth_box(track_id, (x1, y1, x2, y2))

                # Build keypoint dict
                kp_dict = {}
                for name, idx in KP_NAMES.items():
                    x, y = kp_xy[i][idx]
                    if kp_conf is not None:
                        vis = float(kp_conf[i][idx])
                        kp_dict[name] = (float(x), float(y)) if vis > 0.3 else None
                    else:
                        kp_dict[name] = (float(x), float(y))

                # Cache appearance crop for ReID debugging
                bx1, by1, bx2, by2 = box
                h, w = frame.shape[:2]
                crop = frame[max(0,by1):min(h,by2), max(0,bx1):min(w,bx2)]
                if crop.size > 0:
                    self._appearance[track_id] = cv2.resize(crop, (64, 128))

                persons.append({
                    "track_id":   track_id,
                    "box":        box,
                    "confidence": conf,
                    "keypoints":  kp_dict,
                })

        # Clean up history for lost tracks
        active_ids = {p["track_id"] for p in persons}
        self._box_history = {
            k: v for k, v in self._box_history.items() if k in active_ids
        }

        return persons

    def _smooth_box(self, track_id, box):
        """
        Smooth bounding box position using moving average.
        Reduces jitter from frame-to-frame detection noise.
        """
        if track_id not in self._box_history:
            self._box_history[track_id] = []

        self._box_history[track_id].append(box)

        if len(self._box_history[track_id]) > self._SMOOTH_N:
            self._box_history[track_id].pop(0)

        history = self._box_history[track_id]
        smoothed = tuple(
            int(sum(b[i] for b in history) / len(history))
            for i in range(4)
        )
        return smoothed

    @property
    def active_track_ids(self):
        return list(self._box_history.keys())


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  BoT-SORT Tracker Test -- press Q to quit")
    print("  Walk out and back in -- ID should stay more stable")
    print("  Do a pushup -- ID should NOT change")
    print("=" * 60)

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.video_stream import VideoStream

    cam     = VideoStream(source=config.CAMERA_INDEX)
    tracker = PersonTracker()

    # Unique color per track ID
    id_colors = {}
    def get_color(tid):
        if tid not in id_colors:
            import random
            random.seed(tid * 137)
            id_colors[tid] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )
        return id_colors[tid]

    cam.start()

    fps_start  = time.time()
    fps_count  = 0
    fps_disp   = 0.0
    id_changes = 0
    last_ids   = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        fps_count += 1

        # Track -- single call, no separate detect needed
        tracked = tracker.update(frame)

        # Count ID changes (stability metric)
        current_ids = {p["track_id"] for p in tracked}
        new_ids = current_ids - last_ids
        if new_ids and last_ids:
            id_changes += len(new_ids)
        last_ids = current_ids

        # Draw
        for p in tracked:
            x1, y1, x2, y2 = p["box"]
            tid    = p["track_id"]
            color  = get_color(tid)
            kp     = p["keypoints"]

            # Box + ID label
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID {tid}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2)

            # Skeleton
            for a, b in SKELETON:
                if kp.get(a) and kp.get(b):
                    cv2.line(frame,
                             (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])),
                             color, 2)

            # Keypoint dots
            for pt in kp.values():
                if pt:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 4,
                               (0,200,255), -1)

        # FPS
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_disp  = fps_count / elapsed
            fps_count = 0
            fps_start = time.time()

        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS:{fps_disp:.0f}  Tracked:{len(tracked)}  ID changes:{id_changes}",
                    (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, "Do a pushup -- ID should stay stable",
                    (20,65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
        cv2.putText(frame, "Q to quit",
                    (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        cv2.imshow("SmartGym - BoT-SORT Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("SmartGym - BoT-SORT Tracker",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.stop()
    cv2.destroyAllWindows()
    print(f"\nID changes during session: {id_changes}")
    print("Lower number = more stable tracking")
    print("With BoT-SORT this should be much lower than ByteTrack")