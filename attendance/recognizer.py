"""
recognizer.py
-------------
Loads reference face images from a folder and identifies
people inside person bounding boxes using facenet-pytorch.

Uses:
  - MTCNN              : face detector (finds face inside the person crop)
  - InceptionResnetV1  : FaceNet model (creates face embeddings)

Folder structure expected:
    faces/
        kevin.jpeg
        rahul.jpeg

Each filename (without extension) becomes the person's name.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceRecognizer:
    def __init__(self, faces_dir="faces", threshold=0.9):
        """
        faces_dir : folder containing reference face images
        threshold : euclidean distance threshold — lower = stricter (0.7–1.0 typical)
        """
        self.faces_dir = faces_dir
        self.threshold = threshold
        self.known_embeddings = []
        self.known_names = []

        # Use GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Recognizer] Using device: {self.device}")

        # MTCNN — detects and aligns faces
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=40,
            device=self.device,
            keep_all=False   # return only the best face per crop
        )

        # FaceNet — converts face image to 512-d embedding
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        self._load_known_faces()

    def _get_embedding(self, pil_image):
        """Run MTCNN + FaceNet on a PIL image, return embedding or None."""
        face_tensor = self.mtcnn(pil_image)
        if face_tensor is None:
            return None
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding[0].cpu().numpy()

    def _load_known_faces(self):
        """Load and encode all reference images from faces_dir."""
        if not os.path.exists(self.faces_dir):
            raise FileNotFoundError(
                f"[Recognizer] Faces folder not found: '{self.faces_dir}'\n"
                f"  Create it and add one photo per person (e.g. kevin.jpeg)."
            )

        supported = (".jpg", ".jpeg", ".png")
        files = [f for f in os.listdir(self.faces_dir)
                 if f.lower().endswith(supported)]

        if not files:
            raise ValueError(
                f"[Recognizer] No images found in '{self.faces_dir}'.\n"
                f"  Add photos named after each person (e.g. kevin.jpeg)."
            )

        print(f"[Recognizer] Loading {len(files)} reference face(s)...")

        for filename in files:
            # Strip angle suffixes: kevin_left → kevin, kevin_front → kevin
            base = os.path.splitext(filename)[0]
            name = base.split("_")[0].capitalize()
            path = os.path.join(self.faces_dir, filename)

            pil_img = Image.open(path).convert("RGB")
            embedding = self._get_embedding(pil_img)

            if embedding is None:
                print(f"  [!] No face detected in {filename} — skipping.")
                continue

            self.known_embeddings.append(embedding)
            self.known_names.append(name)
            print(f"  ✓ Loaded: {name}")

        print(f"[Recognizer] Ready — known persons: {self.known_names}")

    def identify(self, frame, person_box):
        """
        Crop the person region from a BGR frame and identify the face.
        Returns the name (str) or 'Unknown'.
        """
        x1, y1, x2, y2 = person_box
        crop_bgr = frame[y1:y2, x1:x2]

        if crop_bgr.size == 0:
            return "Unknown"

        # Convert BGR → RGB → PIL
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        embedding = self._get_embedding(pil_crop)
        if embedding is None:
            return "Unknown"

        # Compare against known embeddings using Euclidean distance
        best_name = "Unknown"
        best_dist = float("inf")

        for known_emb, name in zip(self.known_embeddings, self.known_names):
            dist = np.linalg.norm(embedding - known_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist <= self.threshold:
            return best_name

        return "Unknown"

    def draw_label(self, frame, person_box, name, color=None):
        """
        Draw a colored bounding box + name label on the frame.
        Green = known person, Red = Unknown.
        """
        if color is None:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        x1, y1, x2, y2 = person_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Background rectangle for text readability
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, name,
                    (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)

        return frame
    