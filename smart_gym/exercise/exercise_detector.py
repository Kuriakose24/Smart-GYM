"""
exercise/exercise_detector.py  (v4 — side-view + body_angle fix)
-----------------------------------------------------------------
ROOT CAUSE OF OLD BUGS:

BUG 1 — Wrong standing threshold caused "unknown" stuck state
    OLD: knee_avg > 165 AND elbow_avg > 145 for "standing"
    WHY BAD: Real standing knee angle from side view often reads 155-165
             due to slight natural knee bend + YOLOv8 jitter.
             So person stands still → stuck in "unknown" forever.
    FIX: knee_avg > 155 is enough for standing. Added elbow tolerance too.

BUG 2 — Pushup confused with squat / never detected
    OLD: Only used elbow and knee angles.
    WHY BAD: From a side view, during a squat the elbow CAN dip below 120°
             if arms are extended forward (natural squat form). So the old
             code classified a squat as a pushup.
    FIX: Add body_angle check. Pushup REQUIRES body_angle > 50° (horizontal).
         Squat requires body_angle < 40° (upright). This is the definitive
         separator that the old code was completely missing.

BUG 3 — CONFIRM_N = 25 was too slow
    OLD: 25 frames to confirm exercise switch
    WHY BAD: At 15-20 FPS, that's 1.2-1.6 seconds of lag before the
             exercise detector even acknowledges you changed exercise.
             During a fast squat, you'd be halfway done before it confirmed.
    FIX: CONFIRM_N = 10. Fast enough to feel responsive, still stable enough
         to avoid noise-triggered switches.

BUG 4 — Identity key mismatch caused rep counter resets
    OLD: key = name if name != "Unknown" else f"unknown_{track_id}"
    WHY BAD: When BoT-SORT loses track during a pushup (person goes
             horizontal — common!), new track ID is assigned. Person is
             briefly "Unknown" → creates "unknown_9" detector with 0 reps.
             When identity is restored, it uses "Kevin" key which still
             has old reps — but the exercise detector was reset.
    FIX: Keep one detector per KNOWN name (same as RepCounter). For unknowns,
         use track_id as before — but never delete detectors for known names
         (they persist across track ID changes).

DETECTION LOGIC (clear priority order):
    1. PUSHUP  — elbow bent AND body horizontal (body_angle > 50°)
    2. SQUAT   — knee bent AND body upright (body_angle < 45°)
    3. STANDING — both joints near straight
    4. UNKNOWN  — can't tell yet (not enough history)
"""

import sys
import os
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Tuning constants ──────────────────────────────────────────────────────────
# Lower = faster response but more noise-sensitive
# Higher = more stable but sluggish
CONFIRM_N       = 10    # frames needed to confirm an exercise switch (was 25, too slow)
HISTORY_WINDOW  = 10    # rolling average window size


class ExerciseDetector:
    """
    Detects exercise type for ONE person.
    Keyed to person's NAME so it survives track ID changes.
    """

    def __init__(self, name="Unknown"):
        self.name             = name
        self.current          = "unknown"
        self._candidate       = "unknown"
        self._candidate_count = 0

        self._elbow_history = deque(maxlen=HISTORY_WINDOW)
        self._knee_history  = deque(maxlen=HISTORY_WINDOW)
        self._hip_history   = deque(maxlen=HISTORY_WINDOW)
        self._body_history  = deque(maxlen=HISTORY_WINDOW)  # NEW — body inclination

        print(f"[ExerciseDetector] Created for '{name}'")

    def update(self, angles):
        """
        Call every frame with the angles dict from PoseEstimator.
        Returns current confirmed exercise string.
        """
        if angles is None:
            return self.current

        detected = self._detect(angles)

        if detected == self._candidate:
            self._candidate_count += 1
            if self._candidate_count >= CONFIRM_N:
                if detected != self.current:
                    print(f"[ExerciseDetector] {self.name}: "
                          f"'{self.current}' → '{detected}'")
                    self.current = detected
                    # Don't reset count — keep it high so we stay confirmed
        else:
            # New candidate — start counting from 1
            self._candidate       = detected
            self._candidate_count = 1

        return self.current

    def _detect(self, angles):
        """
        Core detection logic using angle histories.
        Uses rolling averages for stability.
        """
        elbow      = angles.get("elbow")
        knee       = angles.get("knee")
        hip        = angles.get("hip")
        body_angle = angles.get("body_angle")   # 0=upright, 90=horizontal

        # Append to histories only if values are available
        if elbow      is not None: self._elbow_history.append(elbow)
        if knee        is not None: self._knee_history.append(knee)
        if hip         is not None: self._hip_history.append(hip)
        if body_angle  is not None: self._body_history.append(body_angle)

        # Need at least 5 frames of data before making any decision
        if len(self._knee_history) < 5:
            return "unknown"

        elbow_avg = sum(self._elbow_history) / len(self._elbow_history) \
                    if self._elbow_history else 180.0
        knee_avg  = sum(self._knee_history)  / len(self._knee_history)
        hip_avg   = sum(self._hip_history)   / len(self._hip_history) \
                    if self._hip_history else 180.0
        body_avg  = sum(self._body_history)  / len(self._body_history) \
                    if self._body_history else 0.0   # default: assume upright

        elbow_var = self._variance(self._elbow_history)
        knee_var  = self._variance(self._knee_history)

        # ── PUSHUP detection ──────────────────────────────────────────────────
        # REQUIRES body to be horizontal — this is what makes it a pushup
        # not a standing curl or squat.
        #
        # body_avg > 50: body tilted more than 50° from vertical = horizontal
        # elbow_avg < 125: elbows are noticeably bent (even at top of pushup)
        # knee_avg > 145: legs are straight (not squatting while doing pushup)
        #
        # Also catches active pushup motion via variance even if not at bottom
        if body_avg > 50:
            if elbow_avg < 125 and knee_avg > 145:
                return "pushup"
            if elbow_var > 20 and knee_avg > 140:
                return "pushup"

        # ── SQUAT detection ───────────────────────────────────────────────────
        # Body is UPRIGHT (body_avg < 45) AND knee is bent
        # This separates squat from pushup definitively.
        #
        # knee_avg < 135: knees are meaningfully bent
        # body_avg < 45: body is mostly upright (not lying down)
        if body_avg < 45:
            if knee_avg < 135:
                return "squat"
            # Active squat motion — variance catches the movement even at mid-range
            if knee_var > 20 and knee_avg < 160:
                return "squat"

        # ── STANDING detection ────────────────────────────────────────────────
        # Both joints near fully extended AND body upright.
        # NOTE: knee 155+ is enough — don't require 165 (too strict, causes
        # "unknown" stuck state with natural knee bend)
        if knee_avg > 155 and elbow_avg > 140 and body_avg < 30:
            return "standing"

        # ── Fallback: standing without body_angle data ─────────────────────
        # If body_angle is unavailable (some keypoints missing), fall back
        # to just joint angles for standing detection
        if len(self._body_history) == 0:
            if knee_avg > 155 and elbow_avg > 140:
                return "standing"

        return "unknown"

    def _variance(self, history):
        if len(history) < 2:
            return 0.0
        mean = sum(history) / len(history)
        return sum((x - mean) ** 2 for x in history) / len(history)

    def get_confidence(self):
        """Returns 0.0 to 1.0 — how confident we are in current exercise."""
        return min(self._candidate_count / CONFIRM_N, 1.0)

    def reset(self):
        """Clear history — call if person leaves frame for a long time."""
        self._elbow_history.clear()
        self._knee_history.clear()
        self._hip_history.clear()
        self._body_history.clear()
        self._candidate       = "unknown"
        self._candidate_count = 0
        # Note: does NOT reset self.current — exercise type persists


class ExerciseDetectorManager:
    """
    Manages one ExerciseDetector per person.

    KEY FIX: Detectors keyed by NAME (same as RepCounter), not track_id.
    This means the exercise detector survives track ID changes — just like
    the rep counter does. Both reset together (never) or persist together.
    """

    def __init__(self, rep_manager=None):
        self.rep_manager = rep_manager
        # KEY FIX: { name_or_fallback: ExerciseDetector }
        # Known persons: keyed by name  → "Kevin"
        # Unknown persons: keyed by track_id → "tid_7"
        self.detectors = {}
        print("[ExerciseDetectorManager] v4 ready — detectors keyed by NAME.")

    def update(self, tracked_with_angles):
        """
        Process all tracked persons this frame.

        tracked_with_angles : list of person dicts, each with:
            "track_id", "name", "angles", "box", "keypoints"

        Returns list with "exercise" and "ex_confidence" added.
        """
        results = []

        for person in tracked_with_angles:
            name     = person.get("name", "Unknown")
            angles   = person.get("angles")
            track_id = person["track_id"]

            # ── Key strategy (THE FIX) ────────────────────────────────────────
            # Known person: use name as key — survives track ID changes
            # Unknown person: use "tid_X" — temporary until identified
            if name != "Unknown":
                key = name
            else:
                key = f"tid_{track_id}"

            # Create detector if new
            if key not in self.detectors:
                self.detectors[key] = ExerciseDetector(name=name)

            detector = self.detectors[key]

            # Update detector's name if we just identified them
            if name != "Unknown" and detector.name != name:
                detector.name = name
                print(f"[ExerciseDetectorManager] Detector identified: '{key}' → '{name}'")

            # Run detection
            exercise = detector.update(angles)

            # Tell rep counter to switch exercise if needed
            if self.rep_manager and name != "Unknown":
                if exercise in ("pushup", "squat"):
                    counter = self.rep_manager.counters.get(name)
                    if counter and counter.exercise != exercise:
                        self.rep_manager.set_exercise_for_person(name, exercise)

            result = dict(person)
            result["exercise"]      = exercise
            result["ex_confidence"] = detector.get_confidence()
            results.append(result)

        return results

    def get_summary(self):
        return {
            key: {"name": d.name, "exercise": d.current, "confidence": d.get_confidence()}
            for key, d in self.detectors.items()
        }

    def migrate_unknown_to_name(self, track_id, name):
        """
        When an unknown person gets identified, migrate their detector
        from "tid_X" key to their actual name.
        This preserves exercise history across the identification moment.
        """
        old_key = f"tid_{track_id}"
        if old_key in self.detectors and name not in self.detectors:
            self.detectors[name] = self.detectors.pop(old_key)
            self.detectors[name].name = name
            print(f"[ExerciseDetectorManager] Migrated detector: '{old_key}' → '{name}'")


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import time
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.video_stream import VideoStream
    from tracking.person_tracker import PersonTracker
    from pose.pose_estimator import PoseEstimator
    from exercise.rep_counter import RepCounterManager

    print("=" * 60)
    print("  Exercise Detector v4 -- press Q to quit")
    print()
    print("  STAND straight  →  GREEN   (standing)")
    print("  DO squats       →  MAGENTA (squat)")
    print("  DO pushups      →  ORANGE  (pushup)")
    print()
    print("  Angles shown: E=elbow  K=knee  B=body_inclination")
    print("  Body > 50° = horizontal = pushup zone")
    print("  Body < 45° = upright    = squat zone")
    print("=" * 60)

    cam       = VideoStream(source=config.CAMERA_INDEX)
    tracker   = PersonTracker()
    estimator = PoseEstimator()
    rep_mgr   = RepCounterManager()
    ex_det    = ExerciseDetectorManager(rep_manager=rep_mgr)

    cam.start()

    fps_start = time.time()
    fps_count = 0
    fps_disp  = 0.0

    EX_COLORS = {
        "pushup":   (0, 165, 255),
        "squat":    (255, 0, 255),
        "standing": (0, 255, 0),
        "unknown":  (100, 100, 100),
    }

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        fps_count += 1
        tracked = tracker.update(frame)

        tracked_with_angles = []
        for person in tracked:
            p = dict(person)
            p["angles"] = estimator.extract(person["keypoints"])
            tracked_with_angles.append(p)

        with_exercise = ex_det.update(tracked_with_angles)
        results       = rep_mgr.update(with_exercise)

        for p in results:
            x1, y1, x2, y2 = p["box"]
            exercise = p.get("exercise", "unknown")
            reps     = p["rep_count"]
            stage    = p["stage"]
            angles   = p.get("angles") or {}
            color    = EX_COLORS.get(exercise, (100, 100, 100))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Skeleton
            kp = p.get("keypoints", {})
            for a, b in [("left_shoulder", "right_shoulder"),
                         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                         ("left_hip", "right_hip"),
                         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                         ("right_hip", "right_knee"), ("right_knee", "right_ankle")]:
                if kp.get(a) and kp.get(b):
                    cv2.line(frame,
                             (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])),
                             color, 2)

            e = angles.get("elbow", 0) or 0
            k = angles.get("knee", 0)  or 0
            b = angles.get("body_angle", 0) or 0

            info = [
                f"{p.get('name', '?')}",
                f"{exercise.upper()}  Reps:{reps}  {stage}",
                f"E:{e:.0f}  K:{k:.0f}  Body:{b:.0f}",
            ]
            for i, line in enumerate(info):
                cv2.putText(frame, line,
                            (x1, y1 - 10 - (len(info) - i) * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, str(reps),
                        (x1 + 5, y1 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_disp  = fps_count / elapsed
            fps_count = 0
            fps_start = time.time()

        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS:{fps_disp:.0f}  v4  body_angle key",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("SmartGym - Exercise Detector v4", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("SmartGym - Exercise Detector v4",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("\nFinal summary:")
    for key, info in ex_det.get_summary().items():
        print(f"  {info['name']}: {info['exercise']}  (confidence: {info['confidence']:.0%})")