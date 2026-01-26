from setup import *
from rtsp_url import *

db = load_db()

# Init InsightFace (RetinaFace + ArcFace)
app = FaceAnalysis( 
        name="buffalo_s", 
        providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        root="./models"
)

app.prepare(ctx_id=0, det_size=(640, 640))

print("InsightFace models:", app.models.keys())

# START
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(rtsp_imou, cv2.CAP_FFMPEG)

cached_faces = []
frame_idx = True

fps = 0.0
frame_count = 0
fps_update_interval = 1.0
fps_timer = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rx1, ry1, rx2, ry2 = get_roi_box(frame)

    # if frame_idx % DETECT_INTERVAL == 0:
    if frame_idx:
        cached_faces = []
        
        roi = frame[ry1:ry2, rx1:rx2]
        faces = app.get(roi)

        # FACE DETECTION = Model RetinaFace built-in inside buffalo_l
        for face in faces:
            # FACE REPRESENTATION = ArcFace built-in inside buffalo_l
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)

            person_id, sim = match_face(emb, db)
            x1, y1, x2, y2 = map(int, face.bbox)

            # frame offset
            x1 += rx1
            x2 += rx1
            y1 += ry1
            y2 += ry1

            # VERIFY LOGIC
            if person_id is not None and sim >= VERIFY_TH:
                label = f"{person_id} {sim:.2f}"
                color = (0,255,0)
                    
            elif person_id is not None and TH_LOW_CONF <= sim < VERIFY_TH:
                label = f"{person_id}? ({sim:.2f})"
                color = (0,165,255) # orange

                face_img = frame[y1:y2, x1:x2]
                ts = int(time.time()*1000)
                save_path = f"{PENDING_DIR}/{person_id}_{sim:.2f}_ts.jpg"
                cv2.imwrite(save_path, face_img)

            else:
                label = "UNKNOWN"
                color = (0,0,255)
            
            cached_faces.append((x1,y1,x2,y2, label, color))

            # # GET FACE LANDMARKS / tạm thời disable ko cần đến khi hiển thị trong hệ thống
            # for (x,y) in face.kps:
            #     cv2.circle(frame, (int(x), int(y)), 2, (0,0,255), -1)
            
            # # ALIGN FACE LANDMARKS
            # aligned_face = align_face(frame, face.kps)
            # frame[368:480,528:640] = aligned_face[0:112,0:112]

    frame_idx = not frame_idx

    # DRAW
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255,255,0), 2)

    for (x1,y1, x2,y2, label, color) in cached_faces:
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # CALCULATE FPS
    frame_count += 1
    now = time.time()

    if now - fps_timer >= fps_update_interval:
        fps = frame_count / (now - fps_timer)
        fps_timer = now
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,0), 2)

    # VIDEO RESULT OUTPUT
    frame = cv2.resize(frame, (960,640))

    cv2.imshow("Face Detection and Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
