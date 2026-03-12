"""
exercise/exercise_detector.py  (v9 — re-entry reset fix)
----------------------------------------------------------
CHANGES vs v8:

FIX 1 — Detector history reset on person re-entry
    PROBLEM: When a known person (e.g. Nikson) leaves the frame and
    re-enters, their ExerciseDetector still exists in self.detectors
    with stale angle history from their previous exercise session.
    If they left doing pushups and re-enter to do squats, the old
    pushup knee/body/elbow values in the 15-frame history window
    dilute the squat signal — the detector keeps outputting "pushup"
    or "unknown" instead of confirming "squat".

    FIX: migrate_unknown_to_name() now always resets the detector:
      - First entry (tid_X → name): reset after migrating
      - Re-entry (tid_X seen but name already exists): reset existing
        named detector and discard the tid_X detector

    The reset() method clears all history deques, last_body_avg,
    candidate state, and exit count — giving a completely fresh
    detection window on every re-entry.

FIX 2 — current exercise state also reset on re-entry
    Previously reset() only cleared history/candidate but left
    self.current intact. On re-entry this meant the detector could
    immediately return the old exercise ("pushup") on the first frame
    before any new history was built, because the hysteresis block
    for "same exercise" fires before the history check.

    reset() now also sets self.current = "unknown" so the person
    must earn their exercise label from scratch each entry.

All v8 fixes preserved:
    ENTER_N=3, history warmup=2, "standing" neutral in hysteresis.
"""
import sys, os
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Tunable constants ─────────────────────────────────────────────────────────
ENTER_N        = 3    # confirm exercise in ~0.2s at 15 FPS
EXIT_N_SQUAT   = 20   # frames of no-signal before exiting squat
EXIT_N_PUSHUP  = 30   # frames of no-signal before exiting pushup
HISTORY_WINDOW = 15


class ExerciseDetector:
    """
    Detects exercise type for ONE person.
    Keyed to person's NAME so it survives track ID changes.
    """

    def __init__(self, name="Unknown"):
        self.name    = name
        self.current = "unknown"

        self._candidate       = "unknown"
        self._candidate_count = 0
        self._exit_count      = 0

        self._elbow_history = deque(maxlen=HISTORY_WINDOW)
        self._knee_history  = deque(maxlen=HISTORY_WINDOW)
        self._hip_history   = deque(maxlen=HISTORY_WINDOW)
        self._body_history  = deque(maxlen=HISTORY_WINDOW)
        self._hdrop_history = deque(maxlen=HISTORY_WINDOW)

        # Last known body_avg — reused when keypoints drop out horizontally
        self._last_body_avg = None

        print(f"[ExerciseDetector] Created for '{name}'")

    @property
    def _exit_n(self):
        return EXIT_N_PUSHUP if self.current == "pushup" else EXIT_N_SQUAT

    def update(self, angles):
        if angles is None:
            return self.current
        detected = self._detect(angles)
        self._apply_hysteresis(detected)
        return self.current

    def _apply_hysteresis(self, detected):
        if self.current == "unknown":
            if detected not in ("unknown", "standing"):
                if detected == self._candidate:
                    self._candidate_count += 1
                    if self._candidate_count >= ENTER_N:
                        print(f"[ExerciseDetector] {self.name}: 'unknown' → '{detected}'")
                        self.current          = detected
                        self._exit_count      = 0
                        self._candidate_count = ENTER_N
                else:
                    self._candidate       = detected
                    self._candidate_count = 1
            # "standing" is neutral — don't reset candidate, don't increment
            # (allows squat candidate to accumulate right after standing signal)

        else:
            exit_n = self._exit_n

            if detected == self.current or detected == "standing":
                self._exit_count = 0

            elif detected == "unknown":
                self._exit_count += 1
                if self._exit_count >= exit_n:
                    print(f"[ExerciseDetector] {self.name}: "
                          f"'{self.current}' → 'unknown' (no signal for {exit_n} frames)")
                    self.current          = "unknown"
                    self._exit_count      = 0
                    self._candidate       = "unknown"
                    self._candidate_count = 0

            else:
                # Different exercise signalled
                if detected != self._candidate:
                    self._candidate       = detected
                    self._candidate_count = 1
                else:
                    self._candidate_count += 1
                self._exit_count += 1

                if self._candidate_count >= ENTER_N and self._exit_count >= exit_n:
                    print(f"[ExerciseDetector] {self.name}: "
                          f"'{self.current}' → '{detected}'")
                    self.current          = detected
                    self._exit_count      = 0
                    self._candidate_count = ENTER_N

    def _detect(self, angles):
        elbow      = angles.get("elbow")
        knee       = angles.get("knee")
        body_angle = angles.get("body_angle")
        is_front   = angles.get("_is_front_view",  False)
        hip_drop   = angles.get("_hip_drop_ratio", None)

        if elbow      is not None: self._elbow_history.append(elbow)
        if knee       is not None: self._knee_history.append(knee)
        if body_angle is not None: self._body_history.append(body_angle)
        if hip_drop   is not None: self._hdrop_history.append(hip_drop)

        if len(self._knee_history) < 2:
            return "unknown"

        elbow_avg = (sum(self._elbow_history) / len(self._elbow_history)
                     if self._elbow_history else 180.0)
        knee_avg  =  sum(self._knee_history)  / len(self._knee_history)
        hdrop_avg = (sum(self._hdrop_history) / len(self._hdrop_history)
                     if self._hdrop_history else None)
        elbow_var = self._variance(self._elbow_history)
        knee_var  = self._variance(self._knee_history)

        # Use last known body_avg when keypoints drop out (pushup horizontal)
        if self._body_history:
            body_avg = sum(self._body_history) / len(self._body_history)
            self._last_body_avg = body_avg
        elif self._last_body_avg is not None:
            body_avg = self._last_body_avg
        else:
            body_avg = 0.0

        # ── PUSHUP ────────────────────────────────────────────────────────────
        if body_avg > config.EXERCISE_PUSHUP_BODY_MIN:
            if elbow_avg < 125 and knee_avg > 145:
                return "pushup"
            if elbow_var > 15 and knee_avg > 140:
                return "pushup"

        # Secondary: confirmed pushup + elbow moving (covers keypoint dropout)
        if self.current == "pushup" and elbow_var > 8:
            return "pushup"

        # ── SQUAT ─────────────────────────────────────────────────────────────
        if body_avg < config.EXERCISE_SQUAT_BODY_MAX:
            if is_front and hdrop_avg is not None and hdrop_avg < 0.65:
                return "squat"
            if knee_avg < 135:
                return "squat"
            if knee_var > 20 and knee_avg < 160:
                return "squat"

        # ── STANDING ──────────────────────────────────────────────────────────
        if (knee_avg  > config.EXERCISE_STANDING_KNEE_MIN
                and elbow_avg > 140
                and body_avg  < 30):
            return "standing"

        if not self._body_history and self._last_body_avg is None:
            if knee_avg > config.EXERCISE_STANDING_KNEE_MIN and elbow_avg > 140:
                return "standing"

        return "unknown"

    def _variance(self, history):
        if len(history) < 2:
            return 0.0
        mean = sum(history) / len(history)
        return sum((x - mean) ** 2 for x in history) / len(history)

    def get_confidence(self):
        return min(self._candidate_count / ENTER_N, 1.0)

    def reset(self):
        """
        Full reset — called on person re-entry so stale history from a
        previous exercise session doesn't bleed into the new session.
        Clears all angle history, resets hysteresis state, AND resets
        self.current to "unknown" so the person must re-earn their label.
        """
        self._elbow_history.clear()
        self._knee_history.clear()
        self._hip_history.clear()
        self._body_history.clear()
        self._hdrop_history.clear()
        self._last_body_avg   = None
        self._candidate       = "unknown"
        self._candidate_count = 0
        self._exit_count      = 0
        self.current          = "unknown"   # FIX 2: was missing — reset exercise label too


class ExerciseDetectorManager:
    def __init__(self, rep_manager=None):
        self.rep_manager = rep_manager
        if rep_manager is None:
            print("[ExerciseDetectorManager] ⚠ rep_manager is None — "
                  "exercise-switch notifications won't reach RepCounter.")
        self.detectors = {}
        print(f"[ExerciseDetectorManager] v9 ready — ENTER_N={ENTER_N}  "
              f"exit squat={EXIT_N_SQUAT}  pushup={EXIT_N_PUSHUP}")

    def update(self, tracked_with_angles):
        results = []
        for person in tracked_with_angles:
            name     = person.get("name", "Unknown")
            angles   = person.get("angles")
            track_id = person["track_id"]

            key = name if name != "Unknown" else f"tid_{track_id}"

            if key not in self.detectors:
                self.detectors[key] = ExerciseDetector(name=name)

            detector = self.detectors[key]

            if name != "Unknown" and detector.name != name:
                detector.name = name

            exercise = detector.update(angles)

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
        Called from main.py every frame once a track ID is identified.

        Two cases handled:

        Case A — First entry (tid_X → name, name not yet in detectors):
            Migrate the tid_X detector to the name key, then reset it.
            The tid_X detector may have been building history as "Unknown"
            which could be from a different person or a partial signal.
            Reset gives a clean slate under the confirmed name.

        Case B — Re-entry (tid_X seen, but name already exists in detectors):
            The person left and came back with a new track ID.
            Their old named detector has stale history from the previous
            exercise session — discard tid_X detector and reset the
            existing named detector so they re-earn their exercise label.
            THIS IS THE MAIN FIX for the Nikson squat-not-counting bug.
        """
        old_key = f"tid_{track_id}"

        if old_key in self.detectors and name not in self.detectors:
            # Case A: first confirmed identity for this track
            self.detectors[name] = self.detectors.pop(old_key)
            self.detectors[name].name = name
            self.detectors[name].reset()
            print(f"[ExerciseDetectorManager] Migrated + reset: '{old_key}' → '{name}'")

        elif old_key in self.detectors and name in self.detectors:
            # Case B: person re-entered with a new track ID
            # Discard the new tid_X detector, reset the existing named one
            del self.detectors[old_key]
            self.detectors[name].reset()
            print(f"[ExerciseDetectorManager] Re-entry detected — reset detector for '{name}'")

        elif name not in self.detectors:
            # Edge case: name appeared without a tid_ entry (shouldn't normally happen)
            self.detectors[name] = ExerciseDetector(name=name)
            print(f"[ExerciseDetectorManager] Created fresh detector for '{name}'")


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import time
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.video_stream      import VideoStream
    from tracking.person_tracker  import PersonTracker
    from pose.pose_estimator      import PoseEstimator
    from exercise.rep_counter     import RepCounterManager

    print("=" * 60)
    print(f"  Exercise Detector v9 -- press Q to quit")
    print(f"  ENTER_N={ENTER_N}  Exit squat={EXIT_N_SQUAT}  pushup={EXIT_N_PUSHUP}")
    print("  Leave frame and re-enter doing a different exercise")
    print("  → new exercise should be detected cleanly")
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
            p["name"]   = "Test"
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

            e    = angles.get("elbow",      0) or 0
            k    = angles.get("knee",       0) or 0
            b    = angles.get("body_angle")
            bstr = f"{b:.0f}" if b is not None else "N/A"
            det    = ex_det.detectors.get("Test")
            exit_c = det._exit_count if det else 0
            exit_n = det._exit_n if det else EXIT_N_SQUAT

            for i, line in enumerate([
                "Test",
                f"{exercise.upper()}  R:{reps}  {stage}",
                f"E:{e:.0f} K:{k:.0f} Body:{bstr}",
                f"exit:{exit_c}/{exit_n}",
            ]):
                cv2.putText(frame, line,
                            (x1, y1 - 10 - (3 - i) * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.putText(frame, str(reps),
                        (x1 + 5, y1 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_disp  = fps_count / elapsed
            fps_count = 0
            fps_start = time.time()

        h, w = frame.shape[:2]
        cv2.putText(frame, f"FPS:{fps_disp:.0f}  v9  ENTER_N={ENTER_N}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("SmartGym - Exercise Detector v9", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("SmartGym - Exercise Detector v9",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("\nFinal summary:")
    for key, info in ex_det.get_summary().items():
        print(f"  {info['name']}: {info['exercise']}  ({info['confidence']:.0%})")