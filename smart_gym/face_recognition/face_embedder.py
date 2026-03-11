"""
face_recognition/face_embedder.py
----------------------------------
Converts a face image into a 512-dimensional embedding vector using FaceNet.
This vector is like a fingerprint — unique to each person.

Two faces of the same person → similar vectors (cosine similarity > 0.70)
Two faces of different people → different vectors (cosine similarity < 0.70)

This file only does ONE thing: face → vector.
Comparing vectors and matching names is done in recognizer.py.

Pipeline:
    BGR frame crop
        ↓
    MTCNN (detects + aligns face to 160x160)
        ↓
    FaceNet / InceptionResnetV1
        ↓
    512-dimensional numpy array
"""

import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import with graceful error message
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except ImportError:
    raise ImportError(
        "[FaceEmbedder] facenet-pytorch not found.\n"
        "  Run: pip install facenet-pytorch --no-deps"
    )


class FaceEmbedder:
    def __init__(self, device=config.FACE_DEVICE):
        self.device = torch.device(device)
        print(f"[FaceEmbedder] Loading FaceNet on {device}...")

        # MTCNN — detects face in image and aligns it to 160x160
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,           # padding around face
            min_face_size=40,    # ignore tiny faces
            keep_all=False,      # return only the best/largest face
            device=self.device,
            post_process=True,   # normalize pixel values
        )

        # FaceNet — converts aligned face to 512-d embedding
        self.facenet = InceptionResnetV1(
            pretrained="vggface2"   # pre-trained on VGGFace2 dataset
        ).eval().to(self.device)

        print("[FaceEmbedder] ✅ FaceNet ready.")

    def get_embedding(self, image):
        """
        Convert an image containing a face into a 512-d embedding.

        image : numpy BGR array (OpenCV frame or crop) OR PIL Image

        Returns:
            numpy array of shape (512,)  — the face embedding
            None                         — if no face detected
        """
        # Convert BGR numpy → PIL RGB (what MTCNN expects)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # MTCNN: detect + align face → 160x160 tensor
        # BUG FIX: MTCNN raises ValueError("torch.cat(): expected a non-empty list")
        # when the crop contains no detectable face candidates (too small, dark,
        # blurry, or partially out of frame). The "if face_tensor is None" guard
        # below never fires in that case because the exception is thrown inside
        # MTCNN before it can return None. Wrap in try/except so a bad crop
        # returns None gracefully instead of crashing the whole pipeline.
        try:
            face_tensor = self.mtcnn(image)
        except (ValueError, RuntimeError):
            return None  # no face found — not a crash, just skip this crop

        if face_tensor is None:
            return None  # no face found in this image

        # Add batch dimension: (160,160,3) → (1,3,160,160)
        face_tensor = face_tensor.unsqueeze(0).to(self.device)

        # FaceNet: face tensor → 512-d embedding
        with torch.no_grad():
            embedding = self.facenet(face_tensor)

        # Return as numpy array, detached from GPU
        return embedding[0].cpu().numpy()   # shape: (512,)

    def get_embedding_from_box(self, frame, box):
        """
        Convenience method — crop a person box from frame and get embedding.
        Smartly detects if box is full-body or face-only and crops accordingly.

        frame : full BGR frame
        box   : (x1, y1, x2, y2) person bounding box
        """
        x1, y1, x2, y2 = map(int, box)
        box_height = y2 - y1
        box_width  = x2 - x1

        # BUG FIX: when person is horizontal (pushup), box_width >> box_height.
        # Cropping the "top 45%" gives chest/floor, not the face.
        # In landscape orientation the face is on one side — we can't reliably
        # locate it from the bounding box alone, so skip recognition entirely.
        # The IdentityLinker holds the last known name, so identity persists.
        if box_width > box_height * 1.3:
            return None  # person is horizontal — face crop unreliable, skip

        # If box is taller than wide → full body visible → crop top 45% for face
        # If box is roughly square or wider → close up / face only → use full box
        if box_height > box_width * 1.2:
            # Full body — crop top 45% = face region
            face_y2 = y1 + int(box_height * 0.45)
        else:
            # Close up / face only — use entire box
            face_y2 = y2

        # Safety clamp to frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        face_y2 = min(h, face_y2)

        crop = frame[y1:face_y2, x1:x2]

        # BUG FIX: min was 20px but MTCNN's min_face_size=40 means anything under
        # 40px will always return no face — just waste time. Match the limit.
        if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
            return None  # crop too small for MTCNN to find anything

        return self.get_embedding(crop)

    @staticmethod
    def cosine_similarity(emb1, emb2):
        """
        Compute cosine similarity between two embeddings.
        Returns float between -1 and 1.
        1.0  = identical
        0.7+ = same person (use config.FACE_SIMILARITY_THRESHOLD)
        0.0  = completely different
        """
        e1 = torch.tensor(emb1).unsqueeze(0)
        e2 = torch.tensor(emb2).unsqueeze(0)
        return float(F.cosine_similarity(e1, e2).item())


# ── Test — run this file directly ─────────────────────────────────────────────
# python face_recognition/face_embedder.py
if __name__ == "__main__":
    import time
    import cv2

    print("=" * 55)
    print("  FaceEmbedder Test — press Q to quit")
    print("  Look at camera — you should see your embedding")
    print("=" * 55)

    embedder = FaceEmbedder()
    cam_source = config.CAMERA_INDEX

    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    print("\n[Test] Camera open — look at the camera...")
    print("[Test] Will capture embedding every 60 frames\n")

    frame_count   = 0
    last_emb      = None
    last_sim      = None
    status_text   = "Waiting for face..."
    status_color  = (200, 200, 200)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Try to get embedding every 60 frames
        if frame_count % 60 == 0:
            t0  = time.time()
            emb = embedder.get_embedding(frame)
            dt  = time.time() - t0

            if emb is not None:
                # Compare with previous embedding (should be very similar)
                if last_emb is not None:
                    sim = FaceEmbedder.cosine_similarity(emb, last_emb)
                    last_sim     = sim
                    status_text  = f"Same person? {'YES' if sim > config.FACE_SIMILARITY_THRESHOLD else 'NO'} (sim={sim:.3f})"
                    status_color = (0, 255, 0) if sim > config.FACE_SIMILARITY_THRESHOLD else (0, 0, 255)
                else:
                    status_text  = f"First embedding captured! ({dt*1000:.0f}ms)"
                    status_color = (0, 255, 255)

                last_emb = emb
                print(f"[Test] Embedding: shape={emb.shape}  "
                      f"min={emb.min():.3f}  max={emb.max():.3f}  "
                      f"time={dt*1000:.0f}ms")
                if last_sim is not None:
                    print(f"[Test] Similarity to last capture: {last_sim:.4f}  "
                          f"({'SAME PERSON ✅' if last_sim > config.FACE_SIMILARITY_THRESHOLD else 'DIFFERENT ❌'})")
            else:
                status_text  = "No face detected — move closer"
                status_color = (0, 80, 255)
                print("[Test] No face detected this frame.")

        # Draw status on frame
        h, w = frame.shape[:2]

        # Dark bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "FaceEmbedder Test",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(frame, status_text,
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Frame {frame_count} — embedding runs every 60 frames",
                    (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        cv2.putText(frame, "Press Q to quit",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)

        cv2.imshow("SmartGym - FaceEmbedder Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        if cv2.getWindowProperty("SmartGym - FaceEmbedder Test",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n✅ FaceEmbedder test complete.")
    if last_emb is not None:
        print(f"   Embedding shape : {last_emb.shape}")
        print(f"   This means FaceNet is working correctly.")
    else:
        print("   ⚠ No embedding captured — make sure your face is visible.")