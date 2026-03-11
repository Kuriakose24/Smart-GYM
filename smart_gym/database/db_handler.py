"""
database/db_handler.py
-----------------------
Supabase database handler for SmartGym.

Tables:
    members     — registered gym members
    attendance  — daily check-ins
    sessions    — exercise sessions (pushup/squat sets)
    form_scores — per-rep form scores and angles

All operations are non-blocking — errors are caught and logged
so a database failure never crashes the main gym pipeline.

Usage:
    db = DBHandler()
    db.mark_attendance("Kevin")
    session_id = db.start_session("Kevin", "pushup")
    db.log_rep(session_id, "Kevin", "pushup", rep_num=1, angles={...}, score=0.85)
    db.end_session(session_id, rep_count=10, best_depth=80, avg_depth=90)
"""

import os
import sys
from datetime import datetime, date

from dotenv import load_dotenv
from supabase import create_client, Client

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env from smart_gym/ directory
_ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), ".env")
load_dotenv(_ENV_PATH)


class DBHandler:
    """
    Handles all Supabase database operations for SmartGym.
    """

    def __init__(self):
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "[DB] SUPABASE_URL or SUPABASE_KEY missing from .env file!\n"
                f"     Expected at: {_ENV_PATH}"
            )

        self.client: Client = create_client(url, key)
        print("[DB] Connected to Supabase.")

        # Cache to avoid duplicate attendance marks
        self._attendance_marked = set()    # { "Kevin_2026-03-10" }

        # Cache active session IDs
        self._active_sessions   = {}       # { "Kevin_pushup": session_id }

    # ── Members ───────────────────────────────────────────────────────────────

    def ensure_member(self, name):
        """
        Add member to members table if not already there.
        Safe to call every time a person is identified.
        """
        if name == "Unknown":
            return
        try:
            self.client.table("members").upsert(
                {"name": name},
                on_conflict="name"
            ).execute()
        except Exception as e:
            print(f"[DB] ensure_member error: {e}")

    def get_all_members(self):
        """Return list of all member names."""
        try:
            res = self.client.table("members").select("name").execute()
            return [r["name"] for r in res.data]
        except Exception as e:
            print(f"[DB] get_all_members error: {e}")
            return []

    # ── Attendance ─────────────────────────────────────────────────────────────

    def mark_attendance(self, name):
        """
        Mark attendance for today. Safe to call every frame —
        uses local cache to avoid hammering the database.

        Returns True if newly marked, False if already marked today.
        """
        if name == "Unknown":
            return False

        today     = date.today().isoformat()
        cache_key = f"{name}_{today}"

        if cache_key in self._attendance_marked:
            return False

        try:
            self.client.table("attendance").upsert(
                {
                    "member_name": name,
                    "date":        today,
                    "time_in":     datetime.now().isoformat(),
                },
                on_conflict="member_name,date"
            ).execute()

            self._attendance_marked.add(cache_key)
            print(f"[DB] Attendance — {name} at "
                  f"{datetime.now().strftime('%H:%M:%S')}")
            self.ensure_member(name)
            return True

        except Exception as e:
            print(f"[DB] mark_attendance error: {e}")
            return False

    def get_attendance_today(self):
        """Return list of members who checked in today."""
        try:
            today = date.today().isoformat()
            res   = self.client.table("attendance")\
                        .select("member_name, time_in")\
                        .eq("date", today)\
                        .execute()
            return res.data
        except Exception as e:
            print(f"[DB] get_attendance_today error: {e}")
            return []

    # ── Sessions ───────────────────────────────────────────────────────────────

    def start_session(self, name, exercise):
        """
        Start a new exercise session.
        Returns session_id (UUID string) or None on error.
        """
        if name == "Unknown":
            return None

        cache_key = f"{name}_{exercise}"

        # Return existing active session
        if cache_key in self._active_sessions:
            return self._active_sessions[cache_key]

        try:
            res = self.client.table("sessions").insert({
                "member_name": name,
                "exercise":    exercise,
                "started_at":  datetime.now().isoformat(),
            }).execute()

            session_id = res.data[0]["id"]
            self._active_sessions[cache_key] = session_id
            print(f"[DB] Session started — {name} {exercise} "
                  f"({session_id[:8]}...)")
            return session_id

        except Exception as e:
            print(f"[DB] start_session error: {e}")
            return None

    def end_session(self, session_id, rep_count, best_depth, avg_depth):
        """Update session with final stats and end timestamp."""
        if session_id is None:
            return
        try:
            self.client.table("sessions").update({
                "rep_count":  rep_count,
                "best_depth": round(best_depth, 1),
                "avg_depth":  round(avg_depth,  1),
                "ended_at":   datetime.now().isoformat(),
            }).eq("id", session_id).execute()

            print(f"[DB] Session ended — {rep_count} reps  "
                  f"best={best_depth:.0f}  avg={avg_depth:.0f}")

            # Remove from cache
            self._active_sessions = {
                k: v for k, v in self._active_sessions.items()
                if v != session_id
            }
        except Exception as e:
            print(f"[DB] end_session error: {e}")

    def update_session_reps(self, session_id, rep_count, best_depth, avg_depth):
        """Update rep count mid-session after every rep."""
        if session_id is None:
            return
        try:
            self.client.table("sessions").update({
                "rep_count":  rep_count,
                "best_depth": round(best_depth, 1),
                "avg_depth":  round(avg_depth,  1),
            }).eq("id", session_id).execute()
        except Exception as e:
            print(f"[DB] update_session_reps error: {e}")

    # ── Form Scores ────────────────────────────────────────────────────────────

    def log_rep(self, session_id, name, exercise,
                rep_number, angles, form_score=None):
        """
        Log a single rep with joint angles and form score.

        angles     : dict with elbow, knee, hip, body keys
        form_score : float 0.0-1.0 from ML model, or None
        """
        if session_id is None or name == "Unknown":
            return
        try:
            self.client.table("form_scores").insert({
                "session_id":  session_id,
                "member_name": name,
                "exercise":    exercise,
                "rep_number":  rep_number,
                "elbow_angle": round(angles.get("elbow", 0), 1),
                "knee_angle":  round(angles.get("knee",  0), 1),
                "hip_angle":   round(angles.get("hip",   0), 1),
                "body_angle":  round(angles.get("body",  0), 1),
                "form_score":  round(form_score, 3) if form_score else None,
                "created_at":  datetime.now().isoformat(),
            }).execute()
        except Exception as e:
            print(f"[DB] log_rep error: {e}")

    # ── Analytics ──────────────────────────────────────────────────────────────

    def get_member_stats(self, name):
        """Return total reps and sessions for a member."""
        try:
            res = self.client.table("sessions")\
                      .select("rep_count, exercise, best_depth")\
                      .eq("member_name", name)\
                      .execute()

            total_reps = sum(r["rep_count"] for r in res.data)
            sessions   = len(res.data)
            best       = min(
                (r["best_depth"] for r in res.data if r["best_depth"] > 0),
                default=0
            )
            return {
                "name":       name,
                "total_reps": total_reps,
                "sessions":   sessions,
                "best_depth": best,
            }
        except Exception as e:
            print(f"[DB] get_member_stats error: {e}")
            return {}

    def get_recent_sessions(self, name, limit=5):
        """Return last N sessions for a member."""
        try:
            res = self.client.table("sessions")\
                      .select("exercise, rep_count, best_depth, started_at")\
                      .eq("member_name", name)\
                      .order("started_at", desc=True)\
                      .limit(limit)\
                      .execute()
            return res.data
        except Exception as e:
            print(f"[DB] get_recent_sessions error: {e}")
            return []


# ── Test ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  DB Handler Test")
    print("=" * 55)

    db = DBHandler()

    # Test 1 — attendance
    print("\n--- Test 1: Attendance ---")
    db.mark_attendance("Kevin")
    db.mark_attendance("Kevin")    # should not duplicate
    db.mark_attendance("Nickson")
    today = db.get_attendance_today()
    print(f"Today: {[r['member_name'] for r in today]}")

    # Test 2 — session
    print("\n--- Test 2: Session ---")
    sid = db.start_session("Kevin", "pushup")
    print(f"Session ID: {sid}")

    # Test 3 — log reps
    print("\n--- Test 3: Log reps ---")
    for i in range(1, 4):
        db.log_rep(
            session_id = sid,
            name       = "Kevin",
            exercise   = "pushup",
            rep_number = i,
            angles     = {"elbow": 85, "knee": 170, "hip": 160, "body": 5},
            form_score = 0.87,
        )
        print(f"  Rep {i} logged")

    # Test 4 — end session
    print("\n--- Test 4: End session ---")
    db.end_session(sid, rep_count=3, best_depth=80, avg_depth=88)

    # Test 5 — stats
    print("\n--- Test 5: Stats ---")
    stats = db.get_member_stats("Kevin")
    print(f"Kevin: {stats}")
    recent = db.get_recent_sessions("Kevin", limit=3)
    print(f"Recent sessions: {recent}")

    print("\n✅ DB test complete!")
    print("   Check Supabase dashboard to verify data.")