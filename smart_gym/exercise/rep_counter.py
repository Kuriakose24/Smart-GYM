"""
exercise/rep_counter.py  (v5 — pushup body_angle dropout fix)
--------------------------------------------------------------
FIXES vs v4:

FIX 1 — body_is_horizontal guard blocks reps when YOLO drops keypoints
    OLD: body_is_horizontal = (body_angle is None) or (body_angle > 40)
    PROBLEM: When horizontal, YOLO often drops shoulder/hip keypoints.
      pose_estimator then returns a LOW body_angle (e.g. 15°) based on
      partial keypoints, or returns None. The "(body_angle is None) → True"
      path worked, but "(body_angle = 15°) > 40 → False" blocked the rep.
    FIX: Track _last_body_angle. If current frame has body_angle=None,
      use last known value. If last known > 40°, permit the rep.
      Also relax: once ExerciseDetector has confirmed "pushup", trust it —
      remove the body_angle guard entirely and just count on elbow angles.
      The ExerciseDetector is already the gate for "is this a pushup?".

FIX 2 — Elbow thresholds slightly loosened for pushup
    PUSHUP_ELBOW_DOWN raised 110 → 115 in config.
    YOLO bilateral average tends to read higher in horizontal position
    because the camera angle flattens the apparent elbow bend.
    115 is more realistic for "arms bent enough to count as down".

FIX 3 — MIN_DOWN_FRAMES reduced 3 → 2
    At ~15 FPS, 3 frames = 0.2s. For fast pushups this is too slow —
    the bottom of the rep might only be held for 1-2 frames before
    coming back up. 2 frames is still enough to filter noise.
"""

import sys
import os
import time
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class AngleSmoother:
    """Rolling average with velocity filter."""
    def __init__(self, window=5, max_velocity=45):
        self.window       = window
        self.max_velocity = max_velocity
        self._history     = deque(maxlen=window)
        self._last        = None

    def update(self, angle):
        if angle is None:
            return self._last
        if self._last is not None:
            velocity = abs(angle - self._last)
            if velocity > self.max_velocity:
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

    HYSTERESIS      = 5.0
    MIN_DOWN_FRAMES = 2     # FIX: reduced from 3 → 2 for faster pushup bottom detection

    SQUAT_MAX_HIP_ANGLE = 170.0

    def __init__(self, name="Unknown", exercise="pushup"):
        self.name     = name
        self.exercise = exercise

        self.rep_count  = 0
        self.stage      = "UP"
        self.feedback   = "Ready — start your exercise!"

        self._elbow_smoother     = AngleSmoother(window=5, max_velocity=45)
        self._knee_smoother      = AngleSmoother(window=5, max_velocity=45)
        self._hip_drop_smoother  = AngleSmoother(window=4, max_velocity=0.3)

        self._down_frames = 0
        self._cooldown    = 0

        # FIX 1: track last known body_angle so dropout frames don't block reps
        self._last_body_angle = None

        self.bottom_angles    = {"elbow": 0.0, "knee": 0.0, "hip": 0.0, "body": 0.0}
        self.best_depth       = 180.0
        self.total_depth      = 0.0
        self._depth_rep_count = 0
        self.last_rep_time    = None
        self.per_exercise_reps = {}

        print(f"[RepCounter] Created for '{name}' — {exercise}")

    def update(self, angles, current_exercise="unknown"):
        if angles is None:
            return False
        if self._cooldown > 0:
            self._cooldown -= 1
            return False
        if current_exercise == "unknown":
            return False
        if current_exercise != self.exercise:
            return False

        # Update last known body angle
        ba = angles.get("body_angle")
        if ba is not None:
            self._last_body_angle = ba

        if self.exercise == "pushup":
            return self._update_pushup(angles)
        elif self.exercise == "squat":
            return self._update_squat(angles)
        return False

    # ── Pushup ────────────────────────────────────────────────────────────────

    def _update_pushup(self, angles):
        raw_elbow  = angles.get("elbow")

        if raw_elbow is None:
            return False

        elbow = self._elbow_smoother.update(raw_elbow)
        if elbow is None:
            return False

        # FIX 1: use last known body_angle instead of current frame value
        # This means keypoint dropout during horizontal position doesn't
        # suddenly make body_angle look like 0° and block rep counting.
        # If ExerciseDetector confirmed "pushup", body WAS horizontal recently.
        body_angle_to_check = angles.get("body_angle") or self._last_body_angle

        # FIX: If ExerciseDetector confirmed pushup, trust it — don't gate on
        # body_angle at all. The exercise detector is already the gate.
        # body_angle guard is only needed if we're NOT yet confirmed.
        # Since we only reach here when current_exercise == "pushup" (confirmed),
        # skip the body_angle guard entirely.
        body_is_horizontal = True   # FIX: was guarded, now always True when confirmed

        up_threshold = config.PUSHUP_ELBOW_UP - self.HYSTERESIS

        if elbow < config.PUSHUP_ELBOW_DOWN and body_is_horizontal:
            self._down_frames += 1
            self.bottom_angles = {
                "elbow": elbow,
                "knee":  angles.get("knee",       0.0) or 0.0,
                "hip":   angles.get("hip",         0.0) or 0.0,
                "body":  body_angle_to_check or 0.0,
            }
            if elbow < self.best_depth:
                self.best_depth = elbow
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Push up!"

        elif elbow > up_threshold:
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    # ── Squat ─────────────────────────────────────────────────────────────────

    def _update_squat(self, angles):
        is_front_view  = angles.get("_is_front_view",  False)
        hip_drop_ratio = angles.get("_hip_drop_ratio", None)

        if is_front_view and hip_drop_ratio is not None:
            return self._update_squat_front_view(angles, hip_drop_ratio)
        else:
            return self._update_squat_side_view(angles)

    def _update_squat_front_view(self, angles, hip_drop_ratio):
        smooth_ratio = self._hip_drop_smoother.update(hip_drop_ratio)
        if smooth_ratio is None:
            return False

        up_threshold = config.SQUAT_HIP_DROP_UP - 0.05

        if smooth_ratio < config.SQUAT_HIP_DROP_DOWN:
            self._down_frames += 1
            self.bottom_angles = {
                "elbow": angles.get("elbow",       0.0) or 0.0,
                "knee":  angles.get("knee",         0.0) or 0.0,
                "hip":   angles.get("hip",           0.0) or 0.0,
                "body":  angles.get("body_angle",    0.0) or 0.0,
            }
            knee_val = angles.get("knee") or 180.0
            if knee_val < self.best_depth:
                self.best_depth = knee_val
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Stand up!"

        elif smooth_ratio > up_threshold:
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    def _update_squat_side_view(self, angles):
        raw_knee  = angles.get("knee")
        hip_angle = angles.get("hip")

        if raw_knee is None:
            return False

        knee = self._knee_smoother.update(raw_knee)
        if knee is None:
            return False

        hip_is_flexing = (hip_angle is None) or (hip_angle < self.SQUAT_MAX_HIP_ANGLE)
        up_threshold   = config.SQUAT_KNEE_UP - self.HYSTERESIS

        if knee < config.SQUAT_KNEE_DOWN and hip_is_flexing:
            self._down_frames += 1
            self.bottom_angles = {
                "elbow": angles.get("elbow",       0.0) or 0.0,
                "knee":  knee,
                "hip":   angles.get("hip",          0.0) or 0.0,
                "body":  angles.get("body_angle",   0.0) or 0.0,
            }
            if knee < self.best_depth:
                self.best_depth = knee
            if self.stage == "UP":
                self.stage    = "DOWN"
                self.feedback = "Stand up!"

        elif knee > up_threshold:
            if self.stage == "DOWN" and self._down_frames >= self.MIN_DOWN_FRAMES:
                self._complete_rep()
                self._down_frames = 0
                return True
            self._down_frames = 0
            if self.stage != "UP":
                self.stage    = "UP"
                self.feedback = f"Rep {self.rep_count} — go again!"

        return False

    # ── Shared ────────────────────────────────────────────────────────────────

    def _complete_rep(self):
        self.rep_count        += 1
        self._depth_rep_count += 1
        self.stage             = "UP"
        self._cooldown         = config.REP_COOLDOWN_FRAMES
        self.last_rep_time     = time.time()
        self.total_depth      += self.bottom_angles.get(
            "elbow" if self.exercise == "pushup" else "knee", 0
        )
        self.per_exercise_reps[self.exercise] = \
            self.per_exercise_reps.get(self.exercise, 0) + 1
        self.feedback = f"Rep {self.rep_count} done!"
        print(f"[RepCounter] ✅ {self.name} — Rep {self.rep_count} "
              f"({self.exercise} #{self.per_exercise_reps[self.exercise]})  "
              f"depth={self.best_depth:.0f}°")

    def get_avg_depth(self):
        if self._depth_rep_count == 0:
            return 0.0
        return self.total_depth / self._depth_rep_count

    def switch_exercise(self, exercise):
        if exercise == self.exercise:
            return
        self.exercise         = exercise
        self.stage            = "UP"
        self.feedback         = "Ready — start your exercise!"
        self._down_frames     = 0
        self._cooldown        = 0
        self.best_depth       = 180.0
        self.total_depth      = 0.0
        self._depth_rep_count = 0
        self._last_body_angle = None
        self._elbow_smoother.reset()
        self._knee_smoother.reset()
        self._hip_drop_smoother.reset()
        print(f"[RepCounter] {self.name} switched to {exercise}")

    def get_stats(self):
        return {
            "name":              self.name,
            "exercise":          self.exercise,
            "reps":              self.rep_count,
            "per_exercise_reps": dict(self.per_exercise_reps),
            "stage":             self.stage,
            "feedback":          self.feedback,
            "best_depth":        round(self.best_depth, 1),
            "avg_depth":         round(self.get_avg_depth(), 1),
        }


class RepCounterManager:
    """Manages one RepCounter per PERSON NAME."""

    def __init__(self, default_exercise="pushup"):
        self.default_exercise = default_exercise
        self.counters = {}
        print(f"[RepCounterManager] v5 ready. Default: {default_exercise}")

    def update(self, tracked_with_angles):
        results = []

        for person in tracked_with_angles:
            name     = person.get("name", "Unknown")
            angles   = person.get("angles")
            exercise = person.get("exercise", "unknown")

            if name == "Unknown":
                result = dict(person)
                result["rep_count"]     = 0
                result["stage"]         = "Identifying..."
                result["feedback"]      = "Please face camera"
                result["rep_completed"] = False
                result["bottom_angles"] = {}
                results.append(result)
                continue

            if name not in self.counters:
                self.counters[name] = RepCounter(
                    name=name,
                    exercise=self.default_exercise
                )

            counter = self.counters[name]
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
    from camera.video_stream        import VideoStream
    from tracking.person_tracker    import PersonTracker
    from pose.pose_estimator        import PoseEstimator
    from exercise.exercise_detector import ExerciseDetectorManager

    print("=" * 60)
    print("  Rep Counter v5 -- press Q to quit")
    print("  P = force pushup  |  S = force squat")
    print("  Pushup body_angle guard removed — ExerciseDetector is the gate")
    print("=" * 60)

    cam       = VideoStream()
    tracker   = PersonTracker()
    estimator = PoseEstimator()
    rep_mgr   = RepCounterManager(default_exercise="pushup")
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
            feedback = p["feedback"]
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
                    cv2.line(frame, (int(kp[a][0]), int(kp[a][1])),
                             (int(kp[b][0]), int(kp[b][1])), color, 2)

            e  = angles.get("elbow",      0) or 0
            k  = angles.get("knee",       0) or 0
            ba = angles.get("body_angle")
            bastr = f"{ba:.0f}" if ba is not None else "N/A"

            for i, line in enumerate([
                "Test",
                f"{exercise.upper()}  R:{reps}  {stage}",
                f"E:{e:.0f}  K:{k:.0f}  Body:{bastr}",
                feedback,
            ]):
                cv2.putText(frame, line,
                            (x1, y1 - 10 - (3 - i) * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

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
        cv2.putText(frame, f"FPS:{fps_disp:.0f}  v5  P=pushup S=squat",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, "Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        cv2.imshow("SmartGym - Rep Counter v5", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("p"):
            rep_mgr.set_exercise_for_all("pushup")
            print("[Test] Forced: pushup")
        elif key == ord("s"):
            rep_mgr.set_exercise_for_all("squat")
            print("[Test] Forced: squat")
        elif cv2.getWindowProperty("SmartGym - Rep Counter v5",
                                   cv2.WND_PROP_VISIBLE) < 1:
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("\nFinal summary:")
    for key, stats in rep_mgr.get_summary().items():
        print(f"  {stats['name']}: {stats['reps']} reps  "
              f"best={stats['best_depth']}°  avg={stats['avg_depth']}°")