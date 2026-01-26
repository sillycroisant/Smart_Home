import cv2
import numpy as np
from insightface.app import FaceAnalysis
import time
import os
from utils import *

# Landmark chuẩn ArcFace (112x112)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041]    # right mouth
], dtype=np.float32)

DB_PATH = "./face_db"
PENDING_DIR = "waiting_4_classify"

# THRESHOLD FOR FACE RECOGNIZE
TH_MATCH = 0.45
TH_NEW = 0.35
TH_UPDATE = 0.55
UPDATE_BOUND = 0.7

VERIFY_TH = 0.5
TH_LOW_CONF = 0.35

DETECT_INTERVAL = 3
ROI_RATIO_W = 0.4
ROI_RATIO_H = 0.6

# ===== FUNCTIONS =====
def align_face(image, landmarks, output_size=(112, 112)):
    """
    image: BGR image (OpenCV)
    landmarks: numpy array shape (5, 2)
    output_size: aligned face size
    """

    src = np.array(landmarks, dtype=np.float32)
    dst = REFERENCE_LANDMARKS.copy()

    if output_size[1] == 128:
        dst[:, 0] += 8.0

    # Similarity transform
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)

    aligned = cv2.warpAffine(
        image,
        M,
        output_size,
        borderValue=0
    )

    return aligned

def cosine_similarity(a, b):
    return np.dot(a,b)

def load_db(db_path="db"):
    db = {}

    for person_id in os.listdir(db_path):
        person_dir = os.path.join(db_path, person_id)

        if not os.path.isdir(person_dir):
            continue

        embs = []
        for file in os.listdir(person_dir):
            if file.startswith("emb_") and file.endswith(".npy"):
                emb = np.load(os.path.join(person_dir, file))
                emb = emb / np.linalg.norm(emb)
                embs.append(emb)

        if len(embs) > 0:
            db[person_id] = np.vstack(embs)

    print(f"[DB] Loaded {len(db)} person/people")
    return db

def save_person(person_id, embedding):
    np.save(os.path.join(DB_PATH, person_id + ".npy"), embedding)

def save_new_person(person_id, embedding, aligned_face):
    person_dir = os.path.join(DB_PATH, person_id)
    os.makedirs(person_dir, exist_ok=True)

    np.save(os.path.join(person_dir, "embeddings.npy"),
            embedding.reshape(1, -1))

    cv2.imwrite(os.path.join(person_dir, "img_001.jpg"),
                aligned_face)

def match_face(embedding, db):  
    best_id = None
    best_sim = -1.0

    for person_id, embs in db.items():
        #mean_emb = embs.mean(axis=0)
        #mean_emb /= np.linalg.norm(mean_emb)

        # sims = cosine_similarity(embedding, mean_emb)
        sims = embs @ embedding # array 12x1 (cosine sim with each embeddings)
        max_sim = np.max(sims)
        if max_sim > best_sim:
            best_sim = max_sim
            best_id = person_id

    return best_id, best_sim

def get_roi_box(frame):
    h, w,_ = frame.shape
    roi_w = int(w * ROI_RATIO_W)
    roi_h = int(h * ROI_RATIO_H)

    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h

    return x1, y1, x2, y2