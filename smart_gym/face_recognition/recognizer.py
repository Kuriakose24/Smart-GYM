"""
face_recognition/recognizer.py
--------------------------------
Loads the face database and identifies people in live frames.

Key design decisions vs your old recognizer.py:
    OLD: Euclidean distance, threshold=0.9, runs EVERY frame → 2 FPS
    NEW: Cosine similarity, threshold=0.65, runs every 30 frames → 20+ FPS

Why cosine similarity instead of euclidean distance?
    Cosine similarity measures the ANGLE between vectors (ignores magnitude).
    This makes it robust to lighting changes and slight pose differences.
    Euclidean distance is affected by overall brightness — cosine is not.

How matching works:
    Each person has 3-5 stored embeddings (different angles).
    We compare the live face against ALL stored embeddings for each person.
    We take the BEST (highest) similarity score per person.
    If best score > threshold → identity confirmed.
"""

import sys
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from face_recognition.face_embedder import FaceEmbedder


class FaceRecognizer:
    def __init__(self,
                 db_path=config.FACE_DB_PATH,
                 threshold=config.FACE_SIMILARITY_THRESHOLD):

        self.threshold = threshold
        self.embedder  = FaceEmbedder()
        self.db        = {}   # { name: [emb1, emb2, ...] }

        self._load_database(db_path)

    def _load_database(self, db_path):
        """Load face embeddings from pickle file."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"[Recognizer] ❌ Database not found: {db_path}\n"
                f"  Run face_recognition/face_database.py first to enroll faces."
            )

        with open(db_path, "rb") as f:
            self.db = pickle.load(f)

        total = sum(len(v) for v in self.db.values())
        print(f"[Recognizer] ✅ Loaded {len(self.db)} person(s), "
              f"{total} embedding(s): {list(self.db.keys())}")

    def identify_from_frame(self, frame, box=None):
        """
        Identify a person from a full frame or a cropped region.

        frame : full BGR frame OR a cropped person region
        box   : optional (x1,y1,x2,y2) — if given, crops face region from frame

        Returns:
            (name, score) e.g. ("Kevin", 0.87)
            ("Unknown", 0.0) if no match found
        """
        # Get embedding
        if box is not None:
            emb = self.embedder.get_embedding_from_box(frame, box)
        else:
            emb = self.embedder.get_embedding(frame)

        if emb is None:
            return "Unknown", 0.0

        return self._match_embedding(emb)

    def _match_embedding(self, query_emb):
        """
        Compare query embedding against all stored embeddings.
        Returns (best_name, best_score).
        """
        best_name  = "Unknown"
        best_score = 0.0

        query_tensor = torch.tensor(query_emb).unsqueeze(0)

        for name, stored_embs in self.db.items():
            for stored_emb in stored_embs:
                stored_tensor = torch.tensor(stored_emb).unsqueeze(0)
                score = float(
                    F.cosine_similarity(query_tensor, stored_tensor).item()
                )
                if score > best_score:
                    best_score = score
                    best_name  = name

        if best_score >= self.threshold:
            return best_name, best_score

        return "Unknown", best_score

    def get_known_names(self):
        """Return list of all enrolled names."""
        return list(self.db.keys())


# ── Test — run this file directly ─────────────────────────────────────────────
# python face_recognition/recognizer.py
if __name__ == "__main__":
    import time
    import cv2

    print("=" * 55)
    print("  Recognizer Test — press Q to quit")
    print("  Stand in front of camera — should show your name")
    print("=" * 55)

    recognizer = FaceRecognizer()
    print(f"\n[Test] Known persons: {recognizer.get_known_names()}")
    print(f"[Test] Threshold: {config.FACE_SIMILARITY_THRESHOLD}")
    print(f"[Test] Recognition runs every {config.FACE_RECOG_EVERY_N_FRAMES} frames\n")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    frame_count  = 0
    last_name    = "Unknown"
    last_score   = 0.0
    recog_time   = 0.0

    fps_start    = time.time()
    fps_count    = 0
    fps_display  = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fps_count   += 1

        # Run recognition every N frames
        if frame_count % config.FACE_RECOG_EVERY_N_FRAMES == 0:
            t0 = time.time()
            last_name, last_score = recognizer.identify_from_frame(frame)
            recog_time = (time.time() - t0) * 1000
            print(f"[Test] Frame {frame_count}: {last_name} "
                  f"(score={last_score:.3f}, time={recog_time:.0f}ms)")

        # FPS
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_start   = time.time()

        # Draw result on frame
        h, w = frame.shape[:2]

        # Color based on result
        if last_name != "Unknown":
            color = (0, 255, 0)      # green = identified
        else:
            color = (0, 0, 255)      # red = unknown

        # Name label — big and centered
        label = f"{last_name}  ({last_score:.2f})"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
        )
        cx = (w - tw) // 2

        # Dark background for text
        cv2.rectangle(frame, (cx - 10, h//2 - 50),
                      (cx + tw + 10, h//2 + 20), (0,0,0), -1)
        cv2.putText(frame, label,
                    (cx, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 2)

        # HUD
        cv2.putText(frame, f"FPS: {fps_display:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Recog every {config.FACE_RECOG_EVERY_N_FRAMES} frames  |  last: {recog_time:.0f}ms",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
        cv2.putText(frame, f"Known: {', '.join(recognizer.get_known_names())}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
        cv2.putText(frame, "Press Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        cv2.imshow("SmartGym - Recognizer Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("SmartGym - Recognizer Test",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Recognizer test complete.")
    print(f"   If your name appeared on screen — face recognition is working!")
    print(f"   If it showed 'Unknown' — try better lighting or add more photos.")