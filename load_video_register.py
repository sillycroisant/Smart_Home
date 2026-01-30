import cv2
import os
import json
import time
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
VIDEO_PATH = "./video/APhu/inside.mp4"   # video hoặc RTSP
PERSON_ID = "APhu"
SAVE_DIR = f"./db/{PERSON_ID}"

MAX_EMB = 12                # số embedding muốn lưu
BLUR_THRESHOLD = 30         # ngưỡng mờ
CAPTURE_INTERVAL = 1     # giãn cách giữa các lần capture (s)

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# INIT MODEL
# =========================
app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(VIDEO_PATH)

# =========================
# UTILS
# =========================
def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# =========================
# REGISTER LOOP
# =========================
embeddings = []
last_capture = 0

print("[INFO] Start registering face from video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    if len(faces) == 1 and len(embeddings) < MAX_EMB:
        face = faces[0]
        x1, y1, x2, y2 = map(int, face.bbox)

        face_img = frame[y1:y2, x1:x2]
        if face_img.size > 0:
            blur = blur_score(face_img)

            if blur > BLUR_THRESHOLD:
                if time.time() - last_capture > CAPTURE_INTERVAL:
                    emb = face.embedding
                    emb = emb / np.linalg.norm(emb)  # normalize
                    embeddings.append(emb)
                    last_capture = time.time()

                    print(f"[CAPTURE] {len(embeddings)}/{MAX_EMB}  blur={int(blur)}")

        # DRAW
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(
            frame,
            f"Captured: {len(embeddings)}",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    cv2.imshow("Register From Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if len(embeddings) >= MAX_EMB:
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# SAVE DATA
# =========================
if len(embeddings) >= 5:
    for i, emb in enumerate(embeddings):
        np.save(f"{SAVE_DIR}/emb_{i:02d}.npy", emb)

    meta = {
        "person_id": PERSON_ID,
        "created_at": datetime.now().isoformat(),
        "num_embeddings": len(embeddings),
        "source": VIDEO_PATH
    }

    with open(f"{SAVE_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[SUCCESS] Face features registered")
else:
    print("[FAIL] Not enough good face samples")
