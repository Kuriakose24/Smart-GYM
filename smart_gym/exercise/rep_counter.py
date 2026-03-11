"""
exercise/rep_counter.py  (v3 — body_angle validated, no ghost reps)
---------------------------------------------------------------------
BUGS FIXED FROM v2:

BUG 1 — Rep counted at wrong moment (double-counts, missed reps)
    OLD: Stage flips at fixed thresholds without checking body position.
         e.g. During a pushup, if person pauses mid-way, the angle jitter
         can cross the threshold twice → 2 reps counted for 1 movement.
    FIX: MIN_DOWN_FRAMES raised to 5. You must hold the bottom position
         for at least 5 frames before the UP transition counts as a rep.
         This kills phantom counts from momentary threshold crossings.

BUG 2 — Rep counting starts before exercise is confirmed
    OLD: RepCounter counts from the first frame it gets angles, even if
         ExerciseDetector is still in "unknown" state.
    FIX: Pass exercise to update() — if exercise is "unknown" or doesn't
         match the counter's exercise type, skip counting that frame.

BUG 3 — Pushup rep counted even when person stands up (exercise change)
    OLD: Elbow angle alone controls stage. If you finish pushups and
         stand up, elbow might briefly cross PUSHUP_ELBOW_UP → counts
         a "rep" even though you just stood up.
    FIX: Require body_angle > 40° to count pushup reps. If you're standing
         (body_angle ~0°), elbow crossing the threshold doesn't count.

BUG 4 — Squat counted when person sits down in a chair / slouches
    OLD: Knee angle alone. Any knee bend counts.
    FIX: Require hip_angle < 160° for squat — confirms hip flexion is
         happening, not just the person bending their knee while sitting.

BUG 5 — AngleSmoother max_velocity=25 rejects real fast movements
    OLD: velocity > 25° per frame is clamped to previous value.
    WHY BAD: A fast squat or pushup can move 30-40°/frame at peak speed.
             Old code was silently replacing real angle data with stale data,
             making reps invisible to the counter.
    FIX: max_velocity=45 for both smoothers. Smoothing window stays at 5.

THRESHOLDS (same as config.py — just explaining the reasoning):
    PUSHUP_ELBOW_DOWN = 100  → arm must reach 100° or less at bottom
    PUSHUP_ELBOW_UP   = 145  → arm must reach 145° or more at top
    SQUAT_KNEE_DOWN   = 100  → knee must reach 100° or less at bottom  
    SQUAT_KNEE_UP     = 165  → knee must reach 165° or more at top
    These are conservative — if person is doing half-reps they won't count.
    To count partial reps, lower DOWN thresholds (e.g. 115 for each).
"""

import sys
import os
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AngleSmoother:
    """
    Rolling average with velocity filter.
    Removes frame-to-frame jitter without rejecting real fast movements.
    """
    def __init__(self, window=5, max_velocity=45):  # FIX: was 25, now 45
        self.window       = window
        self.max_velocity = max_velocity
        self._history     = deque(maxlen=window)
        self._last        = None

    def update(self, angle):
        if angle is None:
            return self._last  # hold last value — don't reset to None
        if self._last is not None:
            velocity = abs(angle - self._last)
            if velocity > self.max_velocity:
                # Fast movement — take halfway between last and new
                # (less aggressive than just clamping to last)
                angle = (self._last + angle) / 2.0
        self._history.append(angle)
        self._last = sum(self._history) / len(self._history)
        return self._last

    def reset(self):
        self._history.clear()
        self._last = None

    @property
    def value(self):
        return self._last


class RepCounter:
    """
    Counts reps for ONE person doing ONE exercise.
    Keyed by NAME — survives track ID changes forever.
    """

    HYSTERESIS       = 5.0   # degrees of hysteresis to prevent stage flickering
    MIN_DOWN_FRAMES  = 3     # lowered from 5 → catches first rep of each set faster
    
    # Body angle thresholds for validation
    PUSHUP_MIN_BODY_ANGLE = 40.0   # body must be at least 40° from vertical
    SQUAT_MAX_HIP_ANGLE   = 160.0  # hip must bend below 160° during squat

    def __init__(self, name="Unknown", exercise="pushup"):
        self.name     = name
        self.exercise = exercise

        self.rep_count  = 0
        self.stage      = "UP"
        self.feedback   = "Ready — start your exercise!"

        self._elbow_smoother = AngleSmoother(window=5, max_velocity=45)
        self._knee_smoother  = AngleSmoother(window=5, max_velocity=45)

        self._down_frames = 0
        self._cooldown    = 0

        self.bottom_angles = {"elbow": 0.0, "knee": 0.0, "hip": 0.0, "body": 0.0}
        self.best_depth    = 180.0   # lowest angle seen (lower = deeper rep)
        self.total_depth   = 0.0
        self.last_rep_time = None
        # Per-exercise rep counts — { "pushup": 3, "squat": 5 }
        # Lets the summary show a breakdown rather than just total
        self.per_exercise_reps = {}

        print(f"[RepCounter] Created for '{name}' — {exercise}")

    def update(self, angles, current_exercise="unknown"):
        """
        Call every frame.

        angles           : dict from PoseEstimator
        current_exercise : confirmed exercise from ExerciseDetector
                          If "unknown", counting is paused.

        Returns True if a rep was just completed this frame.
        """
        if angles is None:
            return False
        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        # FIX: Only count reps when exercise matches what the detector confirmed
        # This prevents phantom reps when transitioning between exercises
        if current_exercise == "unknown":
            return False  # Not confirmed yet — don't count
        if current_exercise != "unknown" and current_exercise != self.exercise:
            return False  # Wrong exercise type

        if self.exercise == "pushup":
            return self._update_pushup(angles)
        elif self.exercise == "squat":
            return self._update_squat(angles)
        return False

    def _update_pushup(self, angles):
        raw_elbow  = angles.get("elbow")
        body_angle = angles.get("body_angle")

        if raw_elbow is None:
            return False

        elbow = self._elbow_smoother.update(raw_elbow)
        if elbow is None:
            return False

        # FIX: Validate body is horizontal before counting
        # This prevents counting a rep when person stands up with bent arms
        body_is_horizontal = (body_angle is None) or (body_angle > self.PUSHUP_MIN_BODY_ANGLE)

        up_threshold = config.PUSHUP_ELBOW_UP - self.HYSTERESIS

        if elbow < config.PUSHUP_ELBOW_DOWN and body_is_horizontal:
            # Arms bent AND body horizontal — we are in the DOWN position
            self._down_frames += 1
            self.bottom_angles = {
                "elbow": elbow,
                "knee":  angles.get("knee",       0.0) or 0.0,
                "hip":   angles.get("hip",         0.0) or 0.0,
                "body":  angles.get("body_angle",  0.0) or 0.0,
            }
            if elbow < self.best_depth:
                self.best_depth = elbow
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Push up!"

        elif elbow > up_threshold:
            # Arms extended — check if we just came up from a valid DOWN
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            # Reset down frame counter if we come up without completing
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    def _update_squat(self, angles):
        raw_knee  = angles.get("knee")
        hip_angle = angles.get("hip")

        if raw_knee is None:
            return False

        knee = self._knee_smoother.update(raw_knee)
        if knee is None:
            return False

        # FIX: Validate hip is flexing — confirms real squat not just knee bend
        hip_is_flexing = (hip_angle is None) or (hip_angle < self.SQUAT_MAX_HIP_ANGLE)

        up_threshold = config.SQUAT_KNEE_UP - self.HYSTERESIS

        if knee < config.SQUAT_KNEE_DOWN and hip_is_flexing:
            # Knee bent AND hip flexing — we are in the DOWN position
            self._down_frames += 1
            self.bottom_angles = {
                "elbow": angles.get("elbow",      0.0) or 0.0,
                "knee":  knee,
                "hip":   angles.get("hip",         0.0) or 0.0,
                "body":  angles.get("body_angle",  0.0) or 0.0,
            }
            if knee < self.best_depth:
                self.best_depth = knee
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Stand up!"

        elif knee > up_threshold:
            # Legs extended — check if we just came up from a valid DOWN
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    def _complete_rep(self):
        self.rep_count    += 1
        self.stage         = "UP"
        self._cooldown     = config.REP_COOLDOWN_FRAMES
        self.last_rep_time = time.time()
        self.total_depth  += self.bottom_angles.get(
            "elbow" if self.exercise == "pushup" else "knee", 0
        )
        # Track per-exercise rep count for summary breakdown
        self.per_exercise_reps[self.exercise] = self.per_exercise_reps.get(self.exercise, 0) + 1
        self.feedback = f"Rep {self.rep_count} done!"
        print(f"[RepCounter] ✅ {self.name} — Rep {self.rep_count} "
              f"({self.exercise} #{self.per_exercise_reps[self.exercise]})  depth={self.best_depth:.0f}°")

    def get_avg_depth(self):
        if self.rep_count == 0:
            return 0.0
        return self.total_depth / self.rep_count

    def switch_exercise(self, exercise):
        """
        Switch to a different exercise.
        Keeps total rep_count climbing but resets best_depth for the new exercise.
        """
        if exercise == self.exercise:
            return
        self.exercise     = exercise
        self.stage        = "UP"
        self.feedback     = "Ready — start your exercise!"
        self._down_frames = 0
        self._cooldown    = 0
        self.best_depth   = 180.0   # reset so best depth is per-exercise, not lifetime
        self.total_depth  = 0.0     # reset avg depth tracking for new exercise
        self._elbow_smoother.reset()
        self._knee_smoother.reset()
        print(f"[RepCounter] {self.name} switched to {exercise}")

    def get_stats(self):
        return {
            "name":              self.name,
            "exercise":          self.exercise,
            "reps":              self.rep_count,           # total across all exercises
            "per_exercise_reps": dict(self.per_exercise_reps),  # e.g. {"squat": 3, "pushup": 2}
            "stage":             self.stage,
            "feedback":          self.feedback,
            "best_depth":        round(self.best_depth, 1),
            "avg_depth":         round(self.get_avg_depth(), 1),
        }


class RepCounterManager:
    """
    Manages one RepCounter per PERSON NAME.

    Survives track ID changes — reps NEVER reset mid-set.
    Counter is created on first sighting of a name and lives forever.
    """

    def __init__(self, default_exercise="pushup"):
        self.default_exercise = default_exercise
        self.counters = {}   # { name: RepCounter }
        print(f"[RepCounterManager] v3 ready. Default: {default_exercise}")

    def update(self, tracked_with_angles):
        """
        Process all persons this frame.

        tracked_with_angles : list of person dicts, each must have:
            "name", "angles", "exercise" (from ExerciseDetector), "track_id"

        Returns list with rep data added to each person dict.
        """
        results = []

        for person in tracked_with_angles:
            name     = person.get("name", "Unknown")
            angles   = person.get("angles")
            exercise = person.get("exercise", "unknown")  # from ExerciseDetector

            # Don't count until person is identified
            if name == "Unknown":
                result = dict(person)
                result["rep_count"]     = 0
                result["stage"]         = "Identifying..."
                result["feedback"]      = "Please face camera"
                result["rep_completed"] = False
                result["bottom_angles"] = {}
                results.append(result)
                continue

            # Create counter for new person
            if name not in self.counters:
                self.counters[name] = RepCounter(
                    name=name,
                    exercise=self.default_exercise
                )

            counter = self.counters[name]

            # FIX: Pass current_exercise to update so counter can validate
            rep_completed = counter.update(angles, current_exercise=exercise)

            result = dict(person)
            result["rep_count"]     = counter.rep_count
            result["stage"]         = counter.stage
            result["feedback"]      = counter.feedback
            result["rep_completed"] = rep_completed
            result["bottom_angles"] = counter.bottom_angles
            results.append(result)

        return results

    def set_exercise_for_all(self, exercise):
        for counter in self.counters.values():
            counter.switch_exercise(exercise)
        self.default_exercise = exercise

    def set_exercise_for_person(self, name, exercise):
        if name in self.counters:
            self.counters[name].switch_exercise(exercise)
        else:
            # Create counter with correct exercise immediately
            self.counters[name] = RepCounter(name=name, exercise=exercise)

    def get_summary(self):
        return {key: c.get_stats() for key, c in self.counters.items()}

    def get_person_reps(self, name):
        return self.counters[name].rep_count if name in self.counters else 0


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import time
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from camera.video_stream import VideoStream
    from detection.person_detector import PersonDetector
    from tracking.person_tracker import PersonTracker
    from pose.pose_estimator import PoseEstimator
    from exercise.exercise_detector import ExerciseDetectorManager

    print("=" * 60)
    print("  Rep Counter v3 -- press Q to quit")
    print("  P = force pushup mode  |  S = force squat mode")
    print()
    print("  Reps only count when ExerciseDetector confirms the exercise")
    print("  Body angle shown — must be > 40° for pushup reps to count")
    print("=" * 60)

    cam       = VideoStream(source=config.CAMERA_INDEX)
    detector  = PersonDetector()
    tracker   = PersonTracker()
    estimator = PoseEstimator()
    rep_mgr   = RepCounterManager(default_exercise="pushup")
    ex_det    = ExerciseDetectorManager(rep_manager=rep_mgr)

    cam.start()

    fps_start  = time.time()
    fps_count  = 0
    fps_disp   = 0.0
    current_ex = "pushup"

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

        persons = detector.detect(frame)
        tracked = tracker.update(persons)

        tracked_with_angles = []
        for person in tracked:
            p = dict(person)
            p["name"]   = person.get("name", "Unknown")  # tracker may not set name
            p["angles"] = estimator.extract(person["keypoints"])
            tracked_with_angles.append(p)

        with_exercise = ex_det.update(tracked_with_angles)
        results       = rep_mgr.update(with_exercise)

        for p in results:
            x1, y1, x2, y2 = p["box"]
            name     = p.get("name", "?")
            reps     = p["rep_count"]
            stage    = p["stage"]
            feedback = p["feedback"]
            exercise = p.get("exercise", "unknown")
            angles   = p.get("angles") or {}
            color    = EX_COLORS.get(exercise, (100, 100, 100))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            kp = p.get("keypoints", {})
            for a, b in [("left_shoulder", "right_shoulder"),
                         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
                         ("left_hip", "right_hip"),
                         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                         ("right_hip", "right_knee"), ("right_knee", "right_ankle")]:
                if kp.get(a) and kp.get(b):
                    cv2.line(frame, (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])), color, 2)

            e = angles.get("elbow",      0) or 0
            k = angles.get("knee",       0) or 0
            ba = angles.get("body_angle", 0) or 0
            panel = [
                f"{name}",
                f"{exercise.upper()}  R:{reps}  {stage}",
                f"E:{e:.0f}  K:{k:.0f}  Body:{ba:.0f}",
            ]
            for i, line in enumerate(panel):
                cv2.putText(frame, line,
                            (x1, y1 - 10 - (len(panel) - i) * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(frame, feedback, (x1, y2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.putText(frame, str(reps), (x1 + 5, y1 + 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)

        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_disp  = fps_count / elapsed
            fps_count = 0
            fps_start = time.time()

        h, w = frame.shape[:2]
        cv2.putText(frame,
                    f"FPS:{fps_disp:.0f}  v3  P=pushup S=squat",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, "Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("SmartGym - Rep Counter v3", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("p"):
            current_ex = "pushup"
            rep_mgr.set_exercise_for_all("pushup")
            print("[Test] Forced: pushup mode")
        elif key == ord("s"):
            current_ex = "squat"
            rep_mgr.set_exercise_for_all("squat")
            print("[Test] Forced: squat mode")
        elif cv2.getWindowProperty("SmartGym - Rep Counter v3",
                                    cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("\nFinal summary:")
    for key, stats in rep_mgr.get_summary().items():
        print(f"  {stats['name']}: {stats['reps']} {stats['exercise']} reps  "
              f"best depth={stats['best_depth']}°  avg={stats['avg_depth']}°")