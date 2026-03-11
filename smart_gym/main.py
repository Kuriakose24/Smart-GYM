"""
main.py — SmartGym AI
----------------------
Wires all modules together into one production pipeline.

Pipeline every frame:
    1. Camera        → raw BGR frame
    2. PersonTracker → BoT-SORT detects + tracks persons (box, keypoints, track_id)
    3. IdentityLinker→ FaceNet identifies each person (adds name)
    4. Attendance    → marks DB once per person per day
    5. PoseEstimator → extracts joint angles from keypoints
    6. ExerciseDetector → auto-detects pushup / squat / standing
    7. RepCounter    → counts reps per person (tied to name)
    8. Database      → logs reps + form scores to Supabase
    9. Display       → draws overlay on frame

Controls:
    Q        — quit
    S        — save snapshot
    R        — reset all rep counters
    SPACE    — pause / resume
"""

import sys
import os
import cv2
import time
import warnings
warnings.filterwarnings("ignore")   # suppress BoT-SORT ReID warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ── Import all modules ────────────────────────────────────────────────────────
import config
from camera.video_stream          import VideoStream
from tracking.person_tracker      import PersonTracker
from identity.identity_linker     import IdentityLinker
from attendance.attendance_tracker import AttendanceTracker
from pose.pose_estimator          import PoseEstimator
from exercise.exercise_detector   import ExerciseDetectorManager
from exercise.rep_counter         import RepCounterManager
from database.db_handler          import DBHandler


# ── Overlay colors ────────────────────────────────────────────────────────────
COLORS = {
    "pushup":   (0, 165, 255),   # orange
    "squat":    (255, 0, 255),   # magenta
    "standing": (0, 255, 0),     # green
    "unknown":  (100, 100, 100), # gray
    "name_bg":  (20, 20, 20),
}

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


def draw_person(frame, person):
    """Draw bounding box, skeleton, name, reps, and feedback for one person."""
    x1, y1, x2, y2 = person["box"]

    h, w = frame.shape[:2]

    # Clamp coordinates inside the frame
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(w, x2))
    y2 = int(min(h, y2))

    # Ignore invalid boxes
    if x1 >= x2 or y1 >= y2:
        return
    
    name     = person.get("name",     "Unknown")
    exercise = person.get("exercise", "unknown")
    reps     = person.get("rep_count", 0)  # total reps across all exercises
    # Show per-exercise reps on screen if available, otherwise total
    counter_obj = None
    per_ex_reps = reps  # default to total
    stage    = person.get("stage",    "UP")
    feedback = person.get("feedback", "")
    angles   = person.get("angles")  or {}
    color    = COLORS.get(exercise, COLORS["unknown"])

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Skeleton
    kp = person.get("keypoints") or {}
    for a, b in SKELETON:
        if kp.get(a) and kp.get(b):
            cv2.line(frame,
                     (int(kp[a][0]), int(kp[a][1])),
                     (int(kp[b][0]), int(kp[b][1])),
                     color, 2)

    # Keypoint dots
    for pt in kp.values():
        if pt:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 200, 255), -1)

    # Name badge
    badge_text = f"{name}"
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), COLORS["name_bg"], -1)
    cv2.putText(frame, badge_text,
                (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Exercise + stage tag — show current exercise rep count separately
    tag = f"{exercise.upper()} | {stage}  [Total:{reps}]"
    cv2.putText(frame, tag,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Feedback text
    if feedback:
        cv2.putText(frame, feedback,
                    (x1, y2 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Big rep counter
    cv2.putText(frame, str(reps),
                (x1 + 6, y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)

    # Angle readouts (small, top right of box)
    # NOTE: angles dict keys always exist but values can be None when keypoints
    # aren't visible — use "or 0" not default arg, since .get("key", 0) still
    # returns None if the key exists with value None.
    elbow = angles.get("elbow") or 0
    knee  = angles.get("knee")  or 0
    cv2.putText(frame, f"E:{elbow:.0f} K:{knee:.0f}",
                (x2 - 90, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def draw_hud(frame, fps, paused, attended_today, persons):
    """Draw top HUD bar and bottom attendance strip."""
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)

    # FPS
    cv2.putText(frame, f"FPS:{fps:.0f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Person count
    cv2.putText(frame, f"People:{len(persons)}",
                (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Controls hint
    cv2.putText(frame, "Q=quit  S=snapshot  R=reset  SPACE=pause",
                (w - 380, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # PAUSED overlay
    if paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "PAUSED",
                    (w//2 - 80, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 4)

    # Bottom attendance strip
    if attended_today:
        strip_y = h - 30
        cv2.rectangle(frame, (0, strip_y - 5), (w, h), (15, 15, 15), -1)
        names_str = "  ✓  ".join(attended_today)
        cv2.putText(frame, f"Today: {names_str}",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 1)


def main():
    print("=" * 60)
    print("  SmartGym AI — Starting up")
    print("=" * 60)

    # ── Init all modules ──────────────────────────────────────────
    cam        = VideoStream(source=config.CAMERA_INDEX)
    tracker    = PersonTracker()
    linker     = IdentityLinker()
    attendance = AttendanceTracker()
    estimator  = PoseEstimator()
    rep_mgr    = RepCounterManager(default_exercise="pushup")
    ex_det     = ExerciseDetectorManager(rep_manager=rep_mgr)

    # Database — optional, system works without it
    db = None
    try:
        db = DBHandler()
    except Exception as e:
        print(f"[Main] DB not available: {e}")
        print("[Main] Continuing without database logging.")

    cam.start()
    print("[Main] Warming up camera...")
    time.sleep(2)
    print("\n[Main] ✅ All modules ready. Starting pipeline...\n")

    # ── State ─────────────────────────────────────────────────────
    fps_start      = time.time()
    fps_count      = 0
    fps_display    = 0.0
    paused         = False
    attended_today = []              # names checked in today
    session_ids    = {}              # { "Kevin_pushup": session_id }
    snapshot_count = 0

    # ── Main loop ─────────────────────────────────────────────────
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("[Main] Camera lost — retrying...")
            time.sleep(0.1)
            continue

        # Debug — print frame color info first 10 frames
        
                # Handle pause
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused
            print(f"[Main] {'Paused' if paused else 'Resumed'}")
        elif key == ord("r"):
            rep_mgr.counters.clear()
            ex_det.detectors.clear()
            print("[Main] Rep counters reset.")
        elif key == ord("s"):
            snapshot_count += 1
            fname = f"snapshot_{snapshot_count:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[Main] Snapshot saved: {fname}")
        elif cv2.getWindowProperty("SmartGym AI", cv2.WND_PROP_VISIBLE) < 1 and fps_count > 30:
            break

        if paused:
            draw_hud(frame, fps_display, paused, attended_today, [])
            cv2.imshow("SmartGym AI", frame)
            continue

        fps_count += 1

        # ── Pipeline ──────────────────────────────────────────────

        # 1. Detect + track (BoT-SORT)
        tracked = tracker.update(frame)

        # 2. Identify (FaceNet)
        identified = linker.update(frame, tracked)

        # 3. Attendance + DB member registration
        for person in identified:
            name = person.get("name", "Unknown")
            if name != "Unknown":
                # Mark CSV attendance
                # Mark CSV attendance
                attendance.update([name])

                # Mark DB attendance
                if db:
                    newly_marked = db.mark_attendance(name)
                    if newly_marked and name not in attended_today:
                        attended_today.append(name)

        # 4. Pose estimation
        with_angles = []
        for person in identified:
            p = dict(person)
            p["angles"] = estimator.extract(person["keypoints"])
            with_angles.append(p)

        # 5. Exercise detection (auto pushup/squat/standing)
        with_exercise = ex_det.update(with_angles)

        for person in with_exercise:
            name = person.get("name", "Unknown")
            exercise = person.get("exercise", "unknown")
            if name != "Unknown" and exercise in ("pushup", "squat"):
                counter = rep_mgr.counters.get(name)
                if counter and counter.exercise != exercise:
                    rep_mgr.set_exercise_for_person(name, exercise)

        # 6. Rep counting
        results = rep_mgr.update(with_exercise)

        # 7. Database logging — reps + form scores
        if db:
            for person in results:
                name     = person.get("name", "Unknown")
                exercise = person.get("exercise", "unknown")
                angles   = person.get("angles") or {}

                if name == "Unknown" or exercise not in ("pushup", "squat"):
                    continue

                # Start session if needed
                sess_key = f"{name}_{exercise}"
                if sess_key not in session_ids:
                    sid = db.start_session(name, exercise)
                    if sid:
                        session_ids[sess_key] = sid

                # Log rep if just completed
                if person.get("rep_completed") and sess_key in session_ids:
                    counter = rep_mgr.counters.get(name)
                    if counter:
                        db.log_rep(
                            session_id = session_ids[sess_key],
                            name       = name,
                            exercise   = exercise,
                            rep_number = counter.rep_count,
                            angles     = person.get("bottom_angles", {}),
                            form_score = None,  # ML model plugs in here
                        )
                        db.update_session_reps(
                            session_id = session_ids[sess_key],
                            rep_count  = counter.rep_count,
                            best_depth = counter.best_depth,
                            avg_depth  = counter.get_avg_depth(),
                        )

        # 8. Draw overlay
        for person in results:
            draw_person(frame, person)

        # 9. HUD
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_start   = time.time()

        draw_hud(frame, fps_display, paused, attended_today, results)

        cv2.namedWindow("SmartGym AI", cv2.WINDOW_NORMAL)
        cv2.imshow("SmartGym AI", frame)

    # ── Shutdown ──────────────────────────────────────────────────
    print("\n[Main] Shutting down...")

    # End all active sessions in DB
    if db:
        for sess_key, sid in session_ids.items():
            name, exercise = sess_key.rsplit("_", 1)
            counter = rep_mgr.counters.get(name)
            if counter:
                db.end_session(
                    sid,
                    rep_count  = counter.rep_count,
                    best_depth = counter.best_depth,
                    avg_depth  = counter.get_avg_depth(),
                )

    cam.stop()
    cv2.destroyAllWindows()

    # Print final summary
    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    summary = rep_mgr.get_summary()
    if summary:
        for key, stats in summary.items():
            per_ex = stats.get("per_exercise_reps", {})
            breakdown = "  ".join(
                f"{ex}: {count}" for ex, count in per_ex.items()
            ) or f"{stats['exercise']}: {stats['reps']}"
            print(f"  {stats['name']:12} | Total: {stats['reps']} reps  ({breakdown})  "
                  f"| best depth: {stats['best_depth']}°")
    else:
        print("  No exercise data recorded.")

    print(f"\n  Attendance today: {attended_today if attended_today else 'None'}")
    print("=" * 60)
    print("  SmartGym AI — Session complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()