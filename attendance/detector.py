"""
detector.py
-----------
Uses YOLOv8 to detect PERSONS in a frame.
Only class 0 (person) is returned — all other objects are ignored.
"""

from ultralytics import YOLO
import cv2


class PersonDetector:
    PERSON_CLASS_ID = 0  # YOLO class 0 = person

    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        """
        model_path : YOLO model file (auto-downloads if not present)
        confidence : minimum detection confidence (0.0 – 1.0)
        """
        self.confidence = confidence
        print(f"[Detector] Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        print("[Detector] YOLO model loaded.")

    def detect(self, frame):
        """
        Run detection on a frame.

        Returns a list of dicts:
            [
                {"box": (x1, y1, x2, y2), "confidence": 0.87},
                ...
            ]
        Only persons above the confidence threshold are returned.
        """
        results = self.model(frame, verbose=False)
        persons = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == self.PERSON_CLASS_ID and conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    persons.append({
                        "box": (x1, y1, x2, y2),
                        "confidence": conf
                    })

        return persons

    def draw_boxes(self, frame, persons, color=(0, 255, 0)):
        """
        Draw raw person bounding boxes on the frame (before recognition).
        Useful for debugging.
        """
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"Person {p['confidence']:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        return frame