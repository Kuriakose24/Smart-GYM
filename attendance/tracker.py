"""
tracker.py
----------
Combines 3 strategies to keep person identity stable
even when face is not visible (turned away, side profile, etc.)

Strategy 1 — Remember last known identity
    Once a person is identified, keep their name until
    the box disappears or timeout is exceeded.

Strategy 2 — Multiple reference photos
    FaceRecognizer already handles this automatically —
    just add kevin_left.jpeg, kevin_right.jpeg to faces/
    and they are all encoded as "Kevin".

Strategy 3 — YOLO box tracking
    Match current boxes to previous boxes by position (IoU).
    If a box overlaps with a box we knew was Kevin,
    keep calling it Kevin even if face recognition fails.

Usage:
    tracker = IdentityTracker(recognizer, timeout=3.0)
    name = tracker.identify(frame, box)
"""

import time
import numpy as np


class IdentityTracker:
    def __init__(self, recognizer, timeout=3.0, iou_threshold=0.1):
        """
        recognizer    : FaceRecognizer instance
        timeout       : seconds before a lost identity resets to Unknown
        iou_threshold : minimum box overlap to count as same person
        """
        self.recognizer    = recognizer
        self.timeout       = timeout
        self.iou_threshold = iou_threshold

        # { track_id: { "name", "box", "last_seen", "confirmed" } }
        self._tracks = {}
        self._next_id = 0

    # ── Public method — call this instead of recognizer.identify() ────────────
    def identify(self, frame, box):
        """
        Identify the person in box using all 3 strategies.
        Returns name (str) or 'Unknown'.
        """
        now = time.time()

        # Step 1: Try face recognition
        name = self.recognizer.identify(frame, box)

        # Step 2: Find matching track by box overlap (IoU)
        matched_id = self._match_track(box)

        if name != "Unknown":
            # ── Face recognised — update or create track ──────
            if matched_id is not None:
                self._tracks[matched_id]["name"]      = name
                self._tracks[matched_id]["box"]       = box
                self._tracks[matched_id]["last_seen"] = now
                self._tracks[matched_id]["confirmed"] = True
            else:
                # New person entering frame
                self._tracks[self._next_id] = {
                    "name":      name,
                    "box":       box,
                    "last_seen": now,
                    "confirmed": True
                }
                self._next_id += 1

        else:
            # ── Face not recognised — use last known identity ──
            if matched_id is not None:
                track = self._tracks[matched_id]
                elapsed = now - track["last_seen"]

                if elapsed <= self.timeout and track["confirmed"]:
                    # Within timeout — keep last known name
                    name = track["name"]
                    # Update box position to follow the person
                    track["box"] = box
                else:
                    # Timeout exceeded — reset to Unknown
                    name = "Unknown"
                    track["confirmed"] = False
            else:
                # No matching track — only assign if exactly one
                # confirmed person exists and one person in frame
                # to avoid mis-assigning identities
                confirmed = [
                    (tid, t) for tid, t in self._tracks.items()
                    if t["confirmed"] and (now - t["last_seen"]) <= self.timeout
                ]
                if len(confirmed) == 1:
                    tid, t = confirmed[0]
                    name = t["name"]
                    self._tracks[tid]["box"] = box

        # Step 3: Clean up stale tracks
        self._cleanup(now)

        return name

    # ── IoU helper — measures overlap between two boxes ──────────────────────
    def _iou(self, box1, box2):
        """
        Intersection over Union between two boxes.
        Higher = more overlap. 1.0 = identical boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _match_track(self, box):
        """
        Find the track whose last known box overlaps most with current box.
        Returns track_id or None.
        """
        best_id  = None
        best_iou = self.iou_threshold  # minimum threshold to count as match

        for track_id, track in self._tracks.items():
            iou = self._iou(box, track["box"])
            if iou > best_iou:
                best_iou = iou
                best_id  = track_id

        return best_id

    def _cleanup(self, now):
        """Remove tracks that have been lost for more than 2x timeout."""
        stale = [tid for tid, t in self._tracks.items()
                 if now - t["last_seen"] > self.timeout * 2]
        for tid in stale:
            del self._tracks[tid]

    def get_active_names(self):
        """Return list of currently active (confirmed) names."""
        now = time.time()
        return [
            t["name"] for t in self._tracks.values()
            if t["confirmed"] and (now - t["last_seen"]) <= self.timeout
        ]