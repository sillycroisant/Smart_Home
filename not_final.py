import cv2
import os
import time
import numpy as np
from collections import defaultdict, Counter

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis

# =========================
# CONFIG
# =========================
VIDEO_PATH = "./video/APhu/inside_1.mp4"
DB_DIR = "./db"
UNKNOWN_DIR = "./unknown"

CONF_THRES = 0.4
PERSON_CLASS_ID = 0

TH_SIM = 0.3    
VOTE_LEN = 5
FACE_MIN_SIZE = 25

COLOR_DETECTING = (0, 165, 255)
COLOR_FINAL  = (0, 255, 0)

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# =========================
# LOAD MODELS
# =========================
yolo = YOLO("./models/yolov8n.pt")

tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.4,
    embedder="mobilenet",
    half=True
)

face_app = FaceAnalysis(name="buffalo_s")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# =========================
# LOAD FACE DATABASE
# =========================
def load_face_db(db_dir):
    db = {}
    for pid in os.listdir(db_dir):
        pdir = os.path.join(db_dir, pid)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for f in os.listdir(pdir):
            if f.endswith(".npy"):
                emb = np.load(os.path.join(pdir, f))
                emb = emb / np.linalg.norm(emb)
                embs.append(emb)
        if embs:
            db[pid] = np.vstack(embs)
    return db

face_db = load_face_db(DB_DIR)

def match_face(emb, db):
    best_id, best_sim = None, 0
    for pid, embs in db.items():
        sims = np.dot(embs, emb)
        sim = sims.max()
        if sim > best_sim:
            best_sim = sim
            best_id = pid
    return best_id, best_sim

# =========================
# MEMORY
# =========================
track_votes = defaultdict(list)    # track_id → [pid,...]
track_final = {}                   # track_id → final label
track_attr = {}
unknown_count = 0

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame, conf=CONF_THRES, classes=[PERSON_CLASS_ID], verbose=False)[0]

    detections = []
    if results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, box)
            detections.append(([x1, y1, x2-x1, y2-y1], score, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        tid = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        person_roi = frame[y1:y2, x1:x2]

        label = "Detecting..."
        color = COLOR_DETECTING

        # =========================
        # FACE RECO (CHỈ KHI CHƯA CHỐT)
        # =========================
        if tid not in track_final:
            faces = face_app.get(person_roi)

            if len(faces) == 1:
                face = faces[0]
                fx1, fy1, fx2, fy2 = map(int, face.bbox)

                if (fx2 - fx1) > FACE_MIN_SIZE:
                    emb = face.embedding
                    emb = emb / np.linalg.norm(emb)

                    pid, sim = match_face(emb, face_db)

                    print(f"[VOTED] Track_ID {tid} sim={sim:.3f}")

                    if pid is not None and sim >= TH_SIM:
                        track_votes[tid].append(pid)
                    else:
                        track_votes[tid].append("UNKNOWN")

            # =========================
            # VOTING
            # =========================
            if len(track_votes[tid]) >= VOTE_LEN:
                final = Counter(track_votes[tid]).most_common(1)[0][0]

                if final == "UNKNOWN":
                    unknown_count += 1
                    final_id = f"UNKNOWN_{unknown_count:01d}"
                    
                    cv2.imwrite(
                        f"{UNKNOWN_DIR}/{final}.jpg",
                        person_roi
                    )
                else:
                    final_id = final

                track_final[tid] = final_id
                track_votes.pop(tid, None)

                # ATTRIBUTES
                if len(faces) == 1:
                    gender = "Male" if face.gender == 1 else "Female"
                    age = int(face.age)
                else:
                    gender, age = "?", "?"

                track_attr[tid] = {
                    "gender": gender,
                    "age": age
                }

                print(f"[INFO] TRACK {tid} → {final_id} ({gender}, {age})")

        # =========================
        # DRAW
        # =========================
        if tid in track_final:
            info = track_attr.get(tid, {})
            label = f"{track_final[tid]} | {info.get('gender','?')} {info.get('age','?')}"
            color = COLOR_FINAL

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 2)

    cv2.imshow("PIPELINE", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
