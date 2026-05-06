from flask import Flask, render_template, request, jsonify
import mysql.connector
import os
import numpy as np
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from werkzeug.utils import secure_filename
from datetime import datetime
from deepface import DeepFace

# ============================================

# ============================================
app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = "static/uploads"
EMBEDDINGS_CACHE_FILE = "face_embeddings_cache.pkl"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ============================================
# GLOBALS: In-memory embeddings cache + lock
# ============================================
embeddings_cache = {}   # { student_id: np.array(embedding) }
cache_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)  # Parallel face comparisons

MODEL_NAME = "Facenet512"   # Fastest accurate model (swap to "ArcFace" if preferred)
DETECTOR_BACKEND = "opencv"  # Fastest detector; use "retinaface" for better accuracy


# ============================================
# DATABASE CONNECTION
# ============================================
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="your password",
        database="db name"
    )


# ============================================
# EMBEDDING CACHE HELPERS
# ============================================
def save_cache_to_disk():
    with cache_lock:
        with open(EMBEDDINGS_CACHE_FILE, "wb") as f:
            pickle.dump(embeddings_cache, f)


def load_cache_from_disk():
    global embeddings_cache
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        with open(EMBEDDINGS_CACHE_FILE, "rb") as f:
            embeddings_cache = pickle.load(f)
        print(f"[Cache] Loaded {len(embeddings_cache)} embeddings from disk.")
    else:
        print("[Cache] No cache file found. Starting fresh.")


def get_embedding(image_path):
    """Extract face embedding using DeepFace. Returns None on failure."""
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        return np.array(result[0]["embedding"])
    except Exception as e:
        print(f"[Embedding Error] {image_path}: {e}")
        return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def build_embeddings_cache():
    """Load all student embeddings into memory on startup."""
    global embeddings_cache
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, face_image_path FROM students WHERE face_image_path IS NOT NULL")
    students = cursor.fetchall()
    cursor.close()
    conn.close()

    new_cache = {}
    for student_id, image_path in students:
        if student_id in embeddings_cache:
            new_cache[student_id] = embeddings_cache[student_id]  # Reuse existing
            continue
        emb = get_embedding(image_path)
        if emb is not None:
            new_cache[student_id] = emb
            print(f"[Cache] Embedded student {student_id}")

    with cache_lock:
        embeddings_cache = new_cache
    save_cache_to_disk()
    print(f"[Cache] Ready with {len(embeddings_cache)} students.")


# ============================================
# FIND BEST MATCH (vectorized cosine similarity)
# ============================================
SIMILARITY_THRESHOLD = 0.72  # Tune: higher = stricter

def find_matching_student(probe_embedding):
    """Compare probe embedding against all cached embeddings. Returns best student_id or None."""
    with cache_lock:
        if not embeddings_cache:
            return None
        ids = list(embeddings_cache.keys())
        matrix = np.stack(list(embeddings_cache.values()))  # shape: (N, D)

    # Vectorized: compute all cosine similarities at once (O(N) single matrix op)
    probe = probe_embedding / (np.linalg.norm(probe_embedding) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normalized = matrix / norms
    similarities = normalized @ probe  # shape: (N,)

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    print(f"[Match] Best score: {best_score:.4f} for student {ids[best_idx]}")

    if best_score >= SIMILARITY_THRESHOLD:
        return ids[best_idx]
    return None


# ============================================
# HOME
# ============================================
@app.route("/")
def home():
    return render_template("index.html")


# ============================================
# REGISTER STUDENT  (also caches new embedding)
# ============================================
@app.route("/register", methods=["POST"])
def register_student():
    try:
        name = request.form.get("name")
        student_code = request.form.get("student_code")
        email = request.form.get("email")
        photo = request.files.get("photo")

        if not name or not student_code:
            return jsonify({"status": "error", "message": "Name and Student ID required"}), 400

        photo_path = None

        if photo and photo.filename != "":
            filename = secure_filename(photo.filename)
            filename = f"{student_code}_{filename}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            photo.save(save_path)
            photo_path = save_path

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO students (full_name, student_code, email, face_image_path)
            VALUES (%s, %s, %s, %s)
        """, (name, student_code, email, photo_path))

        conn.commit()
        new_id = cursor.lastrowid
        cursor.close()
        conn.close()

        # ✅ Cache new student's embedding immediately (non-blocking)
        if photo_path:
            def cache_new():
                emb = get_embedding(photo_path)
                if emb is not None:
                    with cache_lock:
                        embeddings_cache[new_id] = emb
                    save_cache_to_disk()
                    print(f"[Cache] Added embedding for new student {new_id}")
            threading.Thread(target=cache_new, daemon=True).start()

        return jsonify({"status": "success", "message": "Student Registered Successfully!"})

    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================
# MARK ATTENDANCE  (optimized: embedding + vectorized match)
# ============================================
@app.route("/attendance", methods=["POST"])
def mark_attendance():
    try:
        photo = request.files.get("photo")

        if not photo or photo.filename == "":
            return jsonify({"status": "error", "message": "No photo provided"}), 400

        filename = secure_filename(photo.filename)
        save_name = f"attend_{int(datetime.now().timestamp())}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
        photo.save(save_path)

        # Step 1: Get embedding of the incoming face
        probe_embedding = get_embedding(save_path)

        if probe_embedding is None:
            return jsonify({"status": "error", "message": "No face detected in image"}), 400

        # Step 2: Find matching student (vectorized, fast)
        matched_student_id = find_matching_student(probe_embedding)

        conn = get_db_connection()
        cursor = conn.cursor()

        if matched_student_id:
            cursor.execute("SELECT id FROM students WHERE id = %s", (matched_student_id,))
            valid = cursor.fetchone()

            if not valid:
                return jsonify({"status": "error", "message": "Student ID not found in database"})
            now = datetime.now()
            date_str = now.strftime('%Y-%m-%d')
            time_str = now.strftime('%H:%M:%S')

            cursor.execute("""
                INSERT INTO attendance 
                (student_id, attendance_date, attendance_time, status)
                VALUES (%s, %s, %s, 'Present')
                ON DUPLICATE KEY UPDATE
                attendance_time = VALUES(attendance_time),
                status = 'Present'
            """, (matched_student_id, date_str, time_str))

            activity_msg = f"Attendance marked for Student ID: {matched_student_id}"

        else:
            cursor.execute(
                "INSERT INTO unknown_faces (image_path) VALUES (%s)",
                (save_path,)
            )
            activity_msg = "Unknown face detected"

        cursor.execute(
            "INSERT INTO activity_log (message) VALUES (%s)",
            (activity_msg,)
        )

        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({"status": "success", "message": activity_msg})

    except Exception as e:
        print("ATTENDANCE ERROR:", e)
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================
# ADMIN: Rebuild embeddings cache manually
# ============================================
@app.route("/admin/rebuild-cache", methods=["POST"])
def rebuild_cache():
    threading.Thread(target=build_embeddings_cache, daemon=True).start()
    return jsonify({"status": "success", "message": "Cache rebuild started in background."})


# ============================================
# STATS API
# ============================================
@app.route("/api/stats")
def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM attendance WHERE attendance_date = CURDATE()")
    present_today = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM unknown_faces WHERE DATE(detected_at) = CURDATE()")
    unknown_alerts = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    attendance_rate = round((present_today / total_students * 100), 1) if total_students > 0 else 0

    return jsonify({
        "totalStudents": total_students,
        "presentToday": present_today,
        "unknownAlerts": unknown_alerts,
        "attendanceRate": attendance_rate
    })


# ============================================
# ACTIVITY API
# ============================================
@app.route("/api/activity")
def get_activity():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT message, created_at
        FROM activity_log
        ORDER BY created_at DESC
        LIMIT 5
    """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    activity = [{"message": r[0], "time": str(r[1])} for r in rows]
    return jsonify(activity)


# ============================================
# CHART DATA API
# ============================================
@app.route("/api/chart-data")
def chart_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT attendance_date, COUNT(*)
        FROM attendance
        GROUP BY attendance_date
        ORDER BY attendance_date ASC
        LIMIT 7
    """)

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    labels = [str(r[0]) for r in rows]
    values = [r[1] for r in rows]

    return jsonify({"labels": labels, "values": values})


# ============================================
# RUN APP
# ============================================
if __name__ == "__main__":
    load_cache_from_disk()
    build_embeddings_cache()
    app.run(debug=True)