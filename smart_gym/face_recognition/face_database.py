"""
face_recognition/face_database.py
----------------------------------
Manages the face embedding database.
Enrolls people from photos and saves embeddings to a .pkl file.

Enrollment flow:
    1. Put photos in smart_gym/faces/ folder
       Name them: kevin_1.jpg, kevin_2.jpg, kevin_front.jpg
       The part before the FIRST underscore = the person's name
       e.g.  kevin_1.jpg     → "Kevin"
             mom_front.jpg   → "Mom"
             rahul_side.jpg  → "Rahul"

    2. Run:  python face_recognition/face_database.py
       It reads all photos, extracts embeddings, saves to data/face_database.pkl

    3. Done. The recognizer loads this file automatically.

Database structure (saved as pickle):
    {
        "Kevin": [ emb1(512,), emb2(512,), emb3(512,) ],
        "Mom":   [ emb1(512,), emb2(512,) ],
    }

Adding new people later:
    Just drop more photos in faces/ and run this script again.
    Existing people are preserved — only new/updated ones are re-encoded.
"""

import sys
import os
import pickle
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from face_recognition.face_embedder import FaceEmbedder


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceDatabase:
    def __init__(self, db_path=config.FACE_DB_PATH):
        self.db_path  = db_path
        self.embedder = FaceEmbedder()
        self.db       = {}   # { name: [emb1, emb2, ...] }
        self._load()

    # ── Load / Save ───────────────────────────────────────────────────────────
    def _load(self):
        """Load existing database from disk if it exists."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                self.db = pickle.load(f)
            total = sum(len(v) for v in self.db.values())
            print(f"[FaceDB] Loaded existing DB: "
                  f"{len(self.db)} person(s), {total} embedding(s)")
            for name, embs in self.db.items():
                print(f"  → {name}: {len(embs)} photo(s)")
        else:
            print("[FaceDB] No existing database found — will create new one.")

    def save(self):
        """Save database to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)
        print(f"[FaceDB] ✅ Saved to: {self.db_path}")

    # ── Enroll from photos folder ─────────────────────────────────────────────
    def enroll_from_folder(self, folder=config.FACES_DIR, force=False):
        """
        Read all photos from folder, extract embeddings, add to database.

        folder : path to folder containing face photos
        force  : if True, re-encode photos even if person already in DB

        Naming convention:
            kevin_1.jpg      → "Kevin"
            kevin_front.jpg  → "Kevin"
            mom_side.jpg     → "Mom"

        Photos without underscore:
            kevin.jpg        → "Kevin"
        """
        folder = Path(folder)
        if not folder.exists():
            print(f"[FaceDB] ❌ Folder not found: {folder}")
            print(f"  Create it and add photos named like: kevin_1.jpg")
            return

        photo_files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in SUPPORTED_FORMATS
        ]

        if not photo_files:
            print(f"[FaceDB] ❌ No photos found in {folder}")
            print(f"  Add photos like: kevin_1.jpg, kevin_2.jpg, mom_front.jpg")
            return

        print(f"\n[FaceDB] Found {len(photo_files)} photo(s) in {folder}")
        print("-" * 45)

        added   = 0
        skipped = 0
        failed  = 0

        # Group photos by person name
        name_photos = {}
        for photo in sorted(photo_files):
            # Extract name from filename
            stem = photo.stem                      # e.g. "kevin_front"
            name = stem.split("_")[0].capitalize() # e.g. "Kevin"
            name_photos.setdefault(name, []).append(photo)

        for name, photos in name_photos.items():
            print(f"\n  Person: {name} ({len(photos)} photo(s))")

            if name in self.db and not force:
                print(f"    ⚠ Already enrolled — skipping.")
                print(f"    To re-enroll, run with force=True or delete from DB.")
                skipped += len(photos)
                continue

            embeddings = []
            for photo_path in photos:
                img_bgr = cv2.imread(str(photo_path))
                if img_bgr is None:
                    print(f"    ❌ Could not read: {photo_path.name}")
                    failed += 1
                    continue

                emb = self.embedder.get_embedding(img_bgr)

                if emb is None:
                    print(f"    ❌ No face detected in: {photo_path.name}")
                    print(f"       Make sure the photo shows a clear front-facing face")
                    failed += 1
                    continue

                embeddings.append(emb)
                print(f"    ✅ {photo_path.name} → embedding extracted")
                added += 1

            if embeddings:
                self.db[name] = embeddings
                print(f"    ✓ {name} enrolled with {len(embeddings)} embedding(s)")
            else:
                print(f"    ❌ No valid embeddings for {name} — not added to DB")

        print("\n" + "-" * 45)
        print(f"[FaceDB] Enrollment complete:")
        print(f"  Added   : {added} embedding(s)")
        print(f"  Skipped : {skipped} (already in DB)")
        print(f"  Failed  : {failed} (no face detected)")
        print(f"  Total in DB: {len(self.db)} person(s)")

    def remove_person(self, name):
        """Remove a person from the database."""
        name = name.capitalize()
        if name in self.db:
            del self.db[name]
            print(f"[FaceDB] Removed: {name}")
            self.save()
        else:
            print(f"[FaceDB] '{name}' not found in database.")

    def list_persons(self):
        """Print all enrolled persons."""
        if not self.db:
            print("[FaceDB] Database is empty.")
            return
        print(f"\n[FaceDB] Enrolled persons ({len(self.db)}):")
        for name, embs in self.db.items():
            print(f"  → {name}: {len(embs)} embedding(s)")

    def get_all(self):
        """Return the full database dict."""
        return self.db


# ── Run this file directly to enroll faces ────────────────────────────────────
# python face_recognition/face_database.py
if __name__ == "__main__":
    print("=" * 55)
    print("  SmartGym — Face Enrollment")
    print("=" * 55)
    print(f"""
HOW TO ADD PEOPLE:
  1. Take clear face photos (front, slight left, slight right)
  2. Name them like:
       kevin_1.jpg       ← becomes "Kevin"
       kevin_2.jpg       ← becomes "Kevin" (another angle)
       mom_front.jpg     ← becomes "Mom"
       rahul_side.jpg    ← becomes "Rahul"
  3. Put ALL photos in this folder:
       {config.FACES_DIR}
  4. This script will read them all and save to:
       {config.FACE_DB_PATH}

TIPS FOR BEST ACCURACY:
  • Use 3-5 photos per person
  • Include front, slight left turn, slight right turn
  • Good lighting — no shadows on face
  • Face should be clearly visible, not blurry
  • Phone camera photos work great
  • Minimum photo size: 200x200 pixels
""")

    # Check if faces folder has photos
    faces_dir = Path(config.FACES_DIR)
    photos = [f for f in faces_dir.iterdir()
              if f.suffix.lower() in SUPPORTED_FORMATS] if faces_dir.exists() else []

    if not photos:
        print(f"⚠  No photos found in: {config.FACES_DIR}")
        print(f"   Add photos and run this script again.\n")
    else:
        print(f"Found {len(photos)} photo(s) in faces/ folder.")

        # Ask user what to do
        print("\nOptions:")
        print("  [E] Enroll all new people from faces/ folder")
        print("  [F] Force re-enroll everyone (overwrites existing)")
        print("  [L] List currently enrolled people")
        print("  [R] Remove a person from database")
        print("  [Q] Quit")

        while True:
            choice = input("\nChoice: ").strip().upper()

            if choice == "E":
                db = FaceDatabase()
                db.enroll_from_folder(force=False)
                db.save()
                db.list_persons()
                break

            elif choice == "F":
                db = FaceDatabase()
                db.enroll_from_folder(force=True)
                db.save()
                db.list_persons()
                break

            elif choice == "L":
                db = FaceDatabase()
                db.list_persons()

            elif choice == "R":
                db = FaceDatabase()
                db.list_persons()
                name = input("Enter name to remove: ").strip()
                db.remove_person(name)

            elif choice == "Q":
                break

            else:
                print("Invalid choice. Enter E, F, L, R or Q.")