"""
main.py — SmartGym AI  (v2 — full integration fix)
----------------------------------------------------
FIXES vs v1:

FIX 1 — Pipeline order was correct but angles were attached BEFORE identity
    OLD order:  track → identity → angles → exercise → reps  ✓ (correct)
    This was actually fine in the original. Left as-is.

FIX 2 — AttendanceTracker and DBHandler are optional / may not exist
    Old code crashed on startup if attendance/ or database/ modules were
    absent (common when running your half of the project standalone).
    FIX: Both wrapped in try/except with graceful None fallback.
    The pipeline runs fully without them.

FIX 3 — Exercise switch double-notification removed
    Old code notified rep_manager of exercise switches in TWO places:
      (a) inside ExerciseDetectorManager.update() via rep_manager ref
      (b) again in the main loop after ex_det.update()
    This caused switch_exercise() to be called twice on the same frame,
    resetting the smoother and best_depth unnecessarily.
    FIX: Removed the redundant main-loop notification. The ExerciseDetector-
    Manager already handles it internally via its rep_manager reference.

FIX 4 — draw_person shows per-exercise rep count, not just total
    OLD: always showed counter.rep_count (lifetime total across all exercises)
    FIX: Shows per_exercise_reps[exercise] for the current exercise type,
    with the lifetime total in smaller text below.

Pipeline every frame:
    1. VideoStream        → raw BGR frame
    2. PersonTracker      → BoT-SORT detect+track (box, keypoints, track_id)
    3. IdentityLinker     → FaceNet identifies each person (adds name, score)
    4. Attendance         → marks CSV/DB once per person per day [optional]
    5. PoseEstimator      → extracts joint angles from keypoints
    6. ExerciseDetector   → auto-detects pushup / squat / standing
    7. RepCounter         → counts reps per person (tied to name)
    8. Database           → logs reps + form scores [optional]
    9. Display            → draws overlay on frame

Controls:
    Q / ESC  — quit
    SPACE    — pause / resume
    R        — reset all rep counters
    S        — save snapshot
    P        — force pushup mode for all
    K        — force squat mode for all
"""

import sys
import os
import cv2
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from camera.video_stream         import VideoStream
from tracking.person_tracker     import PersonTracker
from identity.identity_linker    import IdentityLinker
from pose.pose_estimator         import PoseEstimator
from exercise.exercise_detector  import ExerciseDetectorManager
from exercise.rep_counter        import RepCounterManager

# Optional modules — system works without them
AttendanceTracker = None
DBHandler         = None

try:
    from attendance.attendance_tracker import AttendanceTracker as _AT
    AttendanceTracker = _AT
except ImportError:
    print("[Main] AttendanceTracker not found — skipping attendance.")

try:
    from database.db_handler import DBHandler as _DB
    DBHandler = _DB
except ImportError:
    print("[Main] DBHandler not found — skipping database logging.")


# ── Display constants ─────────────────────────────────────────────────────────
COLORS = {
    "pushup":   (0, 165, 255),    # orange
    "squat":    (255, 0, 255),    # magenta
    "standing": (0, 255, 0),      # green
    "unknown":  (100, 100, 100),  # gray
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


def draw_person(frame, person, rep_manager=None):
    """Draw bounding box, skeleton, name, reps, and feedback for one person."""
    x1, y1, x2, y2 = person["box"]
    h, w = frame.shape[:2]

    # Clamp coordinates inside frame
    x1 = int(max(0, x1));  y1 = int(max(0, y1))
    x2 = int(min(w, x2));  y2 = int(min(h, y2))
    if x1 >= x2 or y1 >= y2:
        return

    name     = person.get("name",     "Unknown")
    exercise = person.get("exercise", "unknown")
    reps     = person.get("rep_count", 0)      # lifetime total
    stage    = person.get("stage",    "UP")
    feedback = person.get("feedback", "")
    angles   = person.get("angles")   or {}
    color    = COLORS.get(exercise, COLORS["unknown"])

    # FIX: show current-exercise rep count (e.g. pushup:3), not just lifetime total
    per_ex_reps = reps
    if rep_manager and name != "Unknown":
        counter = rep_manager.counters.get(name)
        if counter:
            per_ex_reps = counter.per_exercise_reps.get(exercise, 0)

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Skeleton lines
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
    badge = name
    (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), COLORS["name_bg"], -1)
    cv2.putText(frame, badge, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Exercise + stage tag
    tag = f"{exercise.upper()} | {stage}  [{exercise}:{per_ex_reps}  total:{reps}]"
    cv2.putText(frame, tag, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Feedback text
    if feedback:
        cv2.putText(frame, feedback, (x1, y2 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Big rep number (per-exercise count is more useful while exercising)
    cv2.putText(frame, str(per_ex_reps),
                (x1 + 6, y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 4)

    # Angle readouts
    elbow = angles.get("elbow") or 0
    knee  = angles.get("knee")  or 0
    body  = angles.get("body_angle") or 0
    cv2.putText(frame, f"E:{elbow:.0f} K:{knee:.0f} B:{body:.0f}",
                (x2 - 110, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


def draw_hud(frame, fps, paused, attended_today, persons):
    """Draw top HUD bar and bottom attendance strip."""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)

    cv2.putText(frame, f"FPS:{fps:.0f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"People:{len(persons)}",
                (100, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Q=quit  SPACE=pause  R=reset  S=snap  P=pushup  K=squat",
                (w - 470, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    if paused:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        cv2.putText(frame, "PAUSED",
                    (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 200, 255), 4)

    if attended_today:
        strip_y = h - 30
        cv2.rectangle(frame, (0, strip_y - 5), (w, h), (15, 15, 15), -1)
        names_str = "  ✓  ".join(attended_today)
        cv2.putText(frame, f"Today: {names_str}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 100), 1)


def main():
    print("=" * 60)
    print("  SmartGym AI — Starting up")
    print("=" * 60)

    # ── Init all modules ──────────────────────────────────────────
    cam       = VideoStream(source=config.CAMERA_INDEX)
    tracker   = PersonTracker()
    linker    = IdentityLinker()
    estimator = PoseEstimator()
    rep_mgr   = RepCounterManager(default_exercise="pushup")

    # FIX: always pass rep_manager so exercise switches notify the counter
    ex_det    = ExerciseDetectorManager(rep_manager=rep_mgr)

    # Optional: attendance
    attendance = None
    if AttendanceTracker:
        try:
            attendance = AttendanceTracker()
        except Exception as e:
            print(f"[Main] AttendanceTracker init failed: {e}")

    # Optional: database
    db = None
    if DBHandler:
        try:
            db = DBHandler()
        except Exception as e:
            print(f"[Main] DB not available: {e} — continuing without DB logging.")

    cam.start()
    print("[Main] Warming up camera...")
    time.sleep(2)
    print("\n[Main] ✅ All modules ready. Starting pipeline...\n")

    # ── State ─────────────────────────────────────────────────────
    fps_start      = time.time()
    fps_count      = 0
    fps_display    = 0.0
    paused         = False
    attended_today = []
    session_ids    = {}
    session_stats  = {}
    snapshot_count = 0

    cv2.namedWindow("SmartGym AI", cv2.WINDOW_NORMAL)

    # ── Main loop ─────────────────────────────────────────────────
    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("[Main] Camera lost — retrying...")
            time.sleep(0.1)
            continue

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord(" "):
            paused = not paused
            print(f"[Main] {'Paused' if paused else 'Resumed'}")
        elif key == ord("r"):
            rep_mgr.counters.clear()
            ex_det.detectors.clear()
            print("[Main] Rep counters + detectors reset.")
        elif key == ord("s"):
            snapshot_count += 1
            fname = f"snapshot_{snapshot_count:03d}.jpg"
            cv2.imwrite(fname, frame)
            print(f"[Main] Snapshot saved: {fname}")
        elif key == ord("p"):
            rep_mgr.set_exercise_for_all("pushup")
            print("[Main] Forced: pushup mode for all")
        elif key == ord("k"):
            rep_mgr.set_exercise_for_all("squat")
            print("[Main] Forced: squat mode for all")
        elif (cv2.getWindowProperty("SmartGym AI", cv2.WND_PROP_VISIBLE) < 1
              and fps_count > 30):
            break

        if paused:
            draw_hud(frame, fps_display, paused, attended_today, [])
            cv2.imshow("SmartGym AI", frame)
            continue

        fps_count += 1

        # ── Pipeline ──────────────────────────────────────────────

        # Step 1: Detect + track (BoT-SORT — single call)
        tracked = tracker.update(frame)

        # Step 2: Face identification
        identified = linker.update(frame, tracked)

        # Step 3: Migrate exercise detector keys + mark attendance
        for person in identified:
            name     = person.get("name", "Unknown")
            track_id = person.get("track_id")
            if name != "Unknown":
                # FIX: migrate detector from "tid_X" → name to preserve history
                if track_id is not None:
                    ex_det.migrate_unknown_to_name(track_id, name)

                if attendance:
                    try:
                        attendance.update([name])
                    except Exception:
                        pass

                if db:
                    try:
                        newly_marked = db.mark_attendance(name)
                        if newly_marked and name not in attended_today:
                            attended_today.append(name)
                    except Exception:
                        pass

        # Step 4: Pose estimation — attach angles to every person
        with_angles = []
        for person in identified:
            p = dict(person)
            p["angles"] = estimator.extract(person.get("keypoints") or {})
            with_angles.append(p)

        # Step 5: Exercise detection (auto pushup/squat/standing)
        with_exercise = ex_det.update(with_angles)

        # FIX: exercise-switch notification is handled INSIDE ExerciseDetectorManager
        # via its rep_manager reference. Do NOT duplicate it here — would cause
        # switch_exercise() to fire twice and reset smoothers unnecessarily.

        # Step 6: Rep counting
        results = rep_mgr.update(with_exercise)

        # Step 7: Database logging
        if db:
            for person in results:
                name     = person.get("name", "Unknown")
                exercise = person.get("exercise", "unknown")
                if name == "Unknown" or exercise not in ("pushup", "squat"):
                    continue

                sess_key = f"{name}_{exercise}"
                if sess_key not in session_ids:
                    try:
                        sid = db.start_session(name, exercise)
                        if sid:
                            session_ids[sess_key] = sid
                    except Exception:
                        pass

                if person.get("rep_completed") and sess_key in session_ids:
                    st = session_stats.setdefault(sess_key, {
                        "rep_count":   0,
                        "best_depth":  180.0,
                        "total_depth": 0.0,
                    })
                    st["rep_count"] += 1
                    depth_key = "elbow" if exercise == "pushup" else "knee"
                    rep_depth = person.get("bottom_angles", {}).get(depth_key, 180.0) or 180.0
                    if rep_depth < st["best_depth"]:
                        st["best_depth"] = rep_depth
                    st["total_depth"] += rep_depth

                    try:
                        db.log_rep(
                            session_id = session_ids[sess_key],
                            name       = name,
                            exercise   = exercise,
                            rep_number = st["rep_count"],
                            angles     = person.get("bottom_angles", {}),
                            form_score = None,
                        )
                        db.update_session_reps(
                            session_id = session_ids[sess_key],
                            rep_count  = st["rep_count"],
                            best_depth = st["best_depth"],
                            avg_depth  = st["total_depth"] / st["rep_count"],
                        )
                    except Exception:
                        pass

        # Step 8: Draw overlay
        for person in results:
            draw_person(frame, person, rep_manager=rep_mgr)

        # Step 9: HUD
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = fps_count / elapsed
            fps_count   = 0
            fps_start   = time.time()

        draw_hud(frame, fps_display, paused, attended_today, results)
        cv2.imshow("SmartGym AI", frame)

    # ── Shutdown ──────────────────────────────────────────────────
    print("\n[Main] Shutting down...")

    if db:
        for sess_key, sid in session_ids.items():
            st        = session_stats.get(sess_key, {})
            rep_count = st.get("rep_count",   0)
            best      = st.get("best_depth",  180.0)
            total     = st.get("total_depth", 0.0)
            avg       = (total / rep_count) if rep_count > 0 else 0.0
            try:
                db.end_session(sid, rep_count=rep_count, best_depth=best, avg_depth=avg)
            except Exception:
                pass

    cam.stop()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  Session Summary")
    print("=" * 60)
    summary = rep_mgr.get_summary()
    if summary:
        for key, stats in summary.items():
            per_ex = stats.get("per_exercise_reps", {})
            lines  = []
            for ex, count in per_ex.items():
                sk = f"{stats['name']}_{ex}"
                st = session_stats.get(sk, {})
                bd = st.get("best_depth", 0.0)
                av = ((st["total_depth"] / st["rep_count"])
                      if st.get("rep_count") else 0.0)
                lines.append(f"{ex}: {count} reps  best={bd:.0f}°  avg={av:.0f}°")
            if not lines:
                lines = [f"{stats['exercise']}: {stats['reps']} reps"]
            print(f"  {stats['name']:12} | Total: {stats['reps']} reps")
            for line in lines:
                print(f"    {line}")
    else:
        print("  No exercise data recorded.")

    print(f"\n  Attendance today: {attended_today if attended_today else 'None'}")
    print("=" * 60)
    print("  SmartGym AI — Session complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()