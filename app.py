import os
import cv2
import random
import numpy as np
import threading
from flask import Flask, render_template, Response, jsonify
from collections import deque, Counter

from emotion_utils import (
    detect_faces,
    preprocess_face,
    load_emotion_model,
    emotion_labels
)

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# -------------------------------
# Load Model (Safe Loading)
# -------------------------------
try:
    model = load_emotion_model('model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# -------------------------------
# Global State
# -------------------------------
state = {
    "emotion": "neutral",
    "confidence": 0.0,
    "song": None
}

emotion_buffer = deque(maxlen=10)
history = deque(maxlen=20)

# -------------------------------
# Emotion → Folder Mapping
# -------------------------------
emotion_map = {
    "happy": ["happy", "dance"],
    "sad": ["sad", "lofi"],
    "angry": ["calm"],
    "neutral": ["neutral"],
    "surprise": ["party"],
    "fear": ["calm"],
    "disgust": ["neutral"]
}

# -------------------------------
# Get Random Song
# -------------------------------
def get_random_song(emotion):
    folders = emotion_map.get(emotion, ["neutral"])

    for folder in folders:
        path = os.path.join(app.root_path, 'static', 'songs', folder)
        if os.path.isdir(path):
            songs = [
                f for f in os.listdir(path)
                if f.endswith(('.mp3', '.wav', '.ogg'))
            ]
            if songs:
                return f"/static/songs/{folder}/{random.choice(songs)}"

    return None

# -------------------------------
# Emotion Prediction (Improved)
# -------------------------------
def predict_emotion(face):
    if model is None:
        return "neutral", 0.0

    face = preprocess_face(face)
    preds = model.predict(face, verbose=0)[0]

    # Top-2 logic
    top2_idx = preds.argsort()[-2:][::-1]
    top1, top2 = top2_idx[0], top2_idx[1]

    confidence = float(preds[top1])
    second_conf = float(preds[top2])

    emotion = emotion_labels[top1]

    # Smart fallback
    if confidence < 0.4 or (confidence - second_conf) < 0.15:
        return "neutral", confidence

    return emotion, confidence

# -------------------------------
# Camera Thread (Smooth Performance)
# -------------------------------
cap = cv2.VideoCapture(0)
frame_global = None

def camera_stream():
    global frame_global
    while True:
        success, frame = cap.read()
        if success:
            frame_global = frame

threading.Thread(target=camera_stream, daemon=True).start()

# -------------------------------
# Frame Generator
# -------------------------------
def gen_frames():
    global frame_global

    last_face_rect = None
    frames_without_face = 0

    while True:
        if frame_global is None:
            continue

        frame = frame_global.copy()

        try:
            # Resize for performance
            frame_small = cv2.resize(frame, (640, 480))
            faces, gray = detect_faces(frame_small)

            if len(faces) > 0:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]

                last_face_rect = (x, y, w, h)
                frames_without_face = 0

                face_crop = frame_small[y:y+h, x:x+w]

                emotion, confidence = predict_emotion(face_crop)
                emotion_buffer.append(emotion)

                final_emotion = Counter(emotion_buffer).most_common(1)[0][0]
                history.append(final_emotion)

                # Update song if emotion changes
                if final_emotion != state["emotion"] or state["song"] is None:
                    state["song"] = get_random_song(final_emotion)

                state["emotion"] = final_emotion
                state["confidence"] = confidence

            else:
                frames_without_face += 1

                if frames_without_face > 5:
                    last_face_rect = None
                    state["emotion"] = "neutral"
                    state["confidence"] = 0.0
                    state["song"] = get_random_song("neutral")

        except Exception as e:
            print(f"Prediction error: {e}")
            state["emotion"] = "error"
            state["confidence"] = 0.0

        # Draw face
        if last_face_rect is not None:
            x, y, w, h = last_face_rect
            text = f"{state['emotion']} ({int(state['confidence']*100)}%)"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,255), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )
        x, y, w, h = faces[0]


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    return jsonify({
        "emotion": state["emotion"],
        "confidence": round(state["confidence"], 2),
        "song": state["song"] or "",
        "history": list(history),
        "status": "ok" if state["emotion"] != "error" else "error"
    })

# -------------------------------
# Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)