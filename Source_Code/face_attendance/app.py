from flask import Flask, render_template, request, jsonify
import cv2
import os
import numpy as np
import face_recognition
import pickle
import pandas as pd
from datetime import datetime, timedelta
import base64

app = Flask(__name__)

DATASET_DIR = "dataset"
ENCODINGS_PATH = "encodings/faces.pkl"
ATTENDANCE_FILE = "attendance.xlsx"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
if not os.path.exists("encodings"):
    os.makedirs("encodings")

# Period timings (24-hour format)
PERIOD_TIMINGS = {
    "IT19541 [WEB TECH]": ("08:00", "09:00"),
    "AI19442 [FOML]": ("09:00", "09:50"),
    "AD19651 [BA]": ("10:10", "11:00"),
    "AD19651 [BA]": ("11:00", "12:00"),
    "AD19643 [IDT]": ("12:30", "13:20"),
    "AD19652 [DIS]": ("13:20", "14:10"),
    "LIBRARY": ("17:30", "18:00"),
}

def get_current_period():
    now = datetime.now().time()
    for period, (start, end) in PERIOD_TIMINGS.items():
        start_time = datetime.strptime(start, "%H:%M").time()
        end_time = datetime.strptime(end, "%H:%M").time()
        if start_time <= now <= end_time:
            return period
    return None

def encode_faces():
    encodings = []
    names = []
    for user_folder in os.listdir(DATASET_DIR):
        for img_name in os.listdir(os.path.join(DATASET_DIR, user_folder)):
            img_path = os.path.join(DATASET_DIR, user_folder, img_name)
            img = face_recognition.load_image_file(img_path)
            faces = face_recognition.face_encodings(img)
            if faces:
                encodings.append(faces[0])
                names.append(user_folder)
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump((encodings, names), f)

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route('/dashboard')
def dashboard():
    df = pd.read_excel("attendancee.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])

    today = datetime.today().date()
    yesterday = today - timedelta(days=1)
    last_7_days = today - timedelta(days=6)

    # Daily stats
    today_data = df[df['Date'].dt.date == today]
    yesterday_data = df[df['Date'].dt.date == yesterday]

    today_count = int(today_data['Roll'].nunique())  # Convert to int
    yesterday_count = int(yesterday_data['Roll'].nunique())  # Convert to int
    diff = int(today_count - yesterday_count)  # Convert to int
    total_students = int(df['Roll'].nunique())  # Convert to int

    # Subject-wise attendance today
    period_attendance = today_data.groupby('Period')['Roll'].nunique().to_dict()

    # Weekly attendance trend
    weekly_data = df[df['Date'].dt.date >= last_7_days]
    trend = weekly_data.groupby(df['Date'].dt.date)['Roll'].nunique()
    dates = [(today - timedelta(days=i)).strftime("%d %b") for i in range(6, -1, -1)]
    trend_counts = [int(trend.get(today - timedelta(days=i), 0)) for i in range(6, -1, -1)]  # Convert to int

    return render_template("dash.html",
                           today_count=today_count,
                           yesterday_count=yesterday_count,
                           diff=diff,
                           total_students=total_students,
                           period_attendance=period_attendance,
                           dates=dates,
                           trend_counts=trend_counts)

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data["name"]
    roll = data["roll"]
    count = data["count"]
    image_data = data["image"]

    user_id = f"{name}_{roll}"
    user_path = os.path.join(DATASET_DIR, user_id)
    os.makedirs(user_path, exist_ok=True)

    header, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    cv2.imwrite(f"{user_path}/{count}.jpg", frame)

    if count == 4:  # After 5 images
        encode_faces()

    return None

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

"""@app.route("/mark", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    image_data = data["image"]

    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings, known_names = pickle.load(f)

    header, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    unknown_face_enc = face_recognition.face_encodings(frame)
    if not unknown_face_enc:
        return jsonify({"message": "No face detected"})

    unknown_enc = unknown_face_enc[0]
    matches = face_recognition.compare_faces(known_encodings, unknown_enc)
    face_distances = face_recognition.face_distance(known_encodings, unknown_enc)

    if any(matches):
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]
        full_name, roll = name.split("_")

        today = datetime.today().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")
        period = get_current_period()

        if period is None:
            return jsonify({"message": "Not within class hours."})

        if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
            df = pd.read_excel(ATTENDANCE_FILE, engine='openpyxl')
        else:
            df = pd.DataFrame(columns=["Name", "Roll", "Date", "Period", "Timestamp"])

        already_marked = (
            (df["Name"] == full_name) &
            (df["Roll"] == roll) &
            (df["Date"] == today) &
            (df["Period"] == period)
        ).any()

        if already_marked:
            return jsonify({"message": f"Attendance already marked for Period {period}."})

        new_row = {
            "Name": full_name,
            "Roll": roll,
            "Date": today,
            "Period": period,
            "Timestamp": timestamp
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False)

        return jsonify({"message": f"Attendance marked for {full_name} (Roll: {roll}) - Period {period}"})
    else:
        return jsonify({"message": "No match found"})"""
@app.route("/mark", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    image_data = data["image"]

    with open(ENCODINGS_PATH, "rb") as f:
        known_encodings, known_names = pickle.load(f)

    header, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    unknown_face_enc = face_recognition.face_encodings(frame)
    if not unknown_face_enc:
        return jsonify({"message": "No face detected"})

    unknown_enc = unknown_face_enc[0]
    matches = face_recognition.compare_faces(known_encodings, unknown_enc)
    face_distances = face_recognition.face_distance(known_encodings, unknown_enc)

    if any(matches):
        best_match_index = np.argmin(face_distances)
        name = known_names[best_match_index]
        full_name, roll = name.split("_")

        today = datetime.today().strftime("%Y-%m-%d")
        timestamp = datetime.now().strftime("%H:%M:%S")
        period = get_current_period()

        if period is None:
            return jsonify({"message": "Not within class hours."})

        if os.path.exists(ATTENDANCE_FILE) and os.path.getsize(ATTENDANCE_FILE) > 0:
            df = pd.read_excel(ATTENDANCE_FILE, engine='openpyxl')
        else:
            df = pd.DataFrame(columns=["Name", "Roll", "Date", "Period", "Timestamp"])

        already_marked = (
            (df["Name"] == full_name) &
            (df["Roll"] == roll) &
            (df["Date"] == today) &
            (df["Period"] == period)
        ).any()

        if already_marked:
            return jsonify({"message": f"Attendance already marked for Period {period}."})

        new_row = {
            "Name": full_name,
            "Roll": int(roll),  # Convert Roll to int
            "Date": today,
            "Period": period,
            "Timestamp": timestamp
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(ATTENDANCE_FILE, index=False)

        return jsonify({"message": f"Attendance marked for {full_name} (Roll: {int(roll)}) - Period {period}"})  # Convert Roll to int
    else:
        return jsonify({"message": "No match found"})


if __name__ == "__main__":
    app.run(debug=True)
