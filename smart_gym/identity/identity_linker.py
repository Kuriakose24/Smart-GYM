"""
identity/identity_linker.py  (Production Grade v3)
----------------------------------------------------
Upgrades from v2:
    1. 10-second grace period — known persons kept alive even when
       tracker loses them (covers pushup horizontal position)
    2. Identity carry-over to nearby new track IDs
    3. lost_at timer reset when person reappears
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from face_recognition.recognizer import FaceRecognizer


class IdentityLinker:
    def __init__(self, recognizer=None):
        if recognizer is None:
            self.recognizer = FaceRecognizer()
        else:
            self.recognizer = recognizer

        # { track_id: {"name", "score", "locked", "last_recog", "frame_count", "box", "lost_at"} }
        self.identity_map = {}

        self._frame_num = 0

        print("[IdentityLinker] ✅ Ready.")
        print(f"  Recognition interval : every {config.FACE_RECOG_EVERY_N_FRAMES} frames")
        print(f"  Re-verify interval   : every {config.IDENTITY_TIMEOUT_SECONDS}s")
        print(f"  Threshold            : {config.FACE_SIMILARITY_THRESHOLD}")

    def update(self, frame, tracked_persons):
        self._frame_num += 1
        now = time.time()

        # Get current active track IDs
        active_ids = {p["track_id"] for p in tracked_persons}

        # Handle lost track IDs
        lost_ids = set(self.identity_map.keys()) - active_ids
        for tid in lost_ids:
            name    = self.identity_map[tid]["name"]
            old_box = self.identity_map[tid].get("box")
            locked  = self.identity_map[tid].get("locked", False)
            lost_at = self.identity_map[tid].get("lost_at")

            # For known persons — keep identity alive for 10 seconds
            # This covers pushup horizontal position where tracker loses them
            if locked and name != "Unknown":
                if lost_at is None:
                    # First frame missing — start grace timer
                    self.identity_map[tid]["lost_at"] = now
                    continue  # don't delete yet

                elif now - lost_at < 10.0:
                    # Still within 10s grace period — keep alive
                    continue  # don't delete yet

            # Grace period expired or unknown person — try carry-over then delete
            if name != "Unknown" and old_box is not None:
                best_new_tid = None
                best_dist    = 150

                for p in tracked_persons:
                    new_tid = p["track_id"]
                    if new_tid not in self.identity_map:
                        nx = (p["box"][0] + p["box"][2]) / 2
                        ny = (p["box"][1] + p["box"][3]) / 2
                        ox = (old_box[0] + old_box[2]) / 2
                        oy = (old_box[1] + old_box[3]) / 2
                        dist = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist    = dist
                            best_new_tid = new_tid

                if best_new_tid is not None:
                    self.identity_map[best_new_tid] = dict(self.identity_map[tid])
                    self.identity_map[best_new_tid].pop("lost_at", None)
                    print(f"[IdentityLinker] Carried {name}: "
                          f"ID {tid} → {best_new_tid} (dist={best_dist:.0f}px)")

            del self.identity_map[tid]
            if name != "Unknown":
                print(f"[IdentityLinker] {name} (ID {tid}) left frame.")

        # Process each tracked person
        results = []
        for person in tracked_persons:
            tid = person["track_id"]
            box = person["box"]

            should_run_recog = self._should_run_recognition(tid, now)

            if should_run_recog:
                name, score = self.recognizer.identify_from_frame(frame, box)
                self._update_identity(tid, name, score, now)

                if name != "Unknown":
                    print(f"[IdentityLinker] ID {tid} → {name} "
                          f"(score={score:.3f})")

            # Store current box and reset grace timer
            if tid in self.identity_map:
                self.identity_map[tid]["box"] = box
                self.identity_map[tid].pop("lost_at", None)  # reset grace timer

            # Get current identity
            identity = self.identity_map.get(tid, {})
            name  = identity.get("name",  "Unknown")
            score = identity.get("score", 0.0)

            result = dict(person)
            result["name"]  = name
            result["score"] = score
            results.append(result)

        # Also inject grace-period persons into results so rep counter keeps running
        for tid, identity in self.identity_map.items():
            if tid not in active_ids and identity.get("lost_at") is not None:
                if identity["name"] != "Unknown":
                    # Person is in grace period — inject with last known box
                    result = {
                        "track_id":   tid,
                        "box":        identity.get("box", (0, 0, 1, 1)),
                        "keypoints":  {},
                        "confidence": 0.0,
                        "name":       identity["name"],
                        "score":      identity["score"],
                    }
                    results.append(result)

        return results

    def _should_run_recognition(self, track_id, now):
        if track_id not in self.identity_map:
            self.identity_map[track_id] = {
                "name":        "Unknown",
                "score":       0.0,
                "locked":      False,
                "last_recog":  0,
                "frame_count": 0,
                "box":         None,
                "lost_at":     None,
            }
            return True

        identity = self.identity_map[track_id]
        identity["frame_count"] += 1

        if identity["locked"]:
            time_since = now - identity["last_recog"]
            return time_since >= config.IDENTITY_TIMEOUT_SECONDS
        else:
            return identity["frame_count"] % config.FACE_RECOG_EVERY_N_FRAMES == 0

    def _update_identity(self, track_id, name, score, now):
        identity = self.identity_map[track_id]

        if name != "Unknown":
            identity["name"]       = name
            identity["score"]      = score
            identity["locked"]     = True
            identity["last_recog"] = now
        else:
            if not identity["locked"]:
                identity["name"]       = "Unknown"
                identity["score"]      = score
                identity["last_recog"] = now

    def get_name(self, track_id):
        return self.identity_map.get(track_id, {}).get("name", "Unknown")

    def get_active_identities(self):
        return {
            tid: info["name"]
            for tid, info in self.identity_map.items()
        }

    def get_known_persons(self):
        return [
            info["name"]
            for info in self.identity_map.values()
            if info["name"] != "Unknown"
        ]


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.video_stream import VideoStream
    from tracking.person_tracker import PersonTracker

    print("=" * 58)
    print("  Identity Linker v3 Test — press Q to quit")
    print("  Name should STAY even when face turns away")
    print("  Lie down for pushups — name stays for 10 seconds!")
    print("=" * 58)

    cam     = VideoStream(source=config.CAMERA_INDEX)
    tracker = PersonTracker()
    linker  = IdentityLinker()

    cam.start()

    fps_start   = time.time()
    fps_count   = 0
    fps_display = 0.0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        fps_count += 1

        tracked  = tracker.update(frame)
        identity = linker.update(frame, tracked)

        for p in identity:
            x1, y1, x2, y2 = p["box"]
            name  = p["name"]
            score = p["score"]
            tid   = p["track_id"]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"{name}  ID:{tid}  ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
            cv2.putText(frame, label, (x1+3, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)

            kp = p.get("keypoints", {})
            for a, b in [("left_shoulder","right_shoulder"),
                         ("left_shoulder","left_elbow"),("left_elbow","left_wrist"),
                         ("right_shoulder","right_elbow"),("right_elbow","right_wrist"),
                         ("left_shoulder","left_hip"),("right_shoulder","right_hip"),
                         ("left_hip","right_hip"),
                         ("left_hip","left_knee"),("left_knee","left_ankle"),
                         ("right_hip","right_knee"),("right_knee","right_ankle")]:
                if kp.get(a) and kp.get(b):
                    cv2.line(frame,
                             (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])),
                             color, 2)

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_start   = time.time()

        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS:{fps_display:.1f}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        known = linker.get_known_persons()
        cv2.putText(frame,
                    f"Identified: {', '.join(known) if known else 'none yet...'}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(frame, "Q to quit",
                    (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        cv2.imshow("SmartGym - Identity Linker v3", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("SmartGym - Identity Linker v3",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.stop()
    cv2.destroyAllWindows()
    print(f"\n✅ Test complete.")
    print(f"   Final: {linker.get_active_identities()}")