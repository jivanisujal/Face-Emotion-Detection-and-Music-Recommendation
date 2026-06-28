import cv2
import numpy as np
from keras.models import load_model
from collections import deque, Counter
import os
from emotion_utils import emotion_labels

# 1. Configuration & Model Loading

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Select model path (fallback logic)
MODEL_PATH = 'best_model.h5' if os.path.exists('best_model.h5') else 'model.h5'
print(f"Loading emotion model from: {MODEL_PATH}")

emotion_model = load_model(MODEL_PATH, compile=False)

# Emotion → Music mapping
emotion_to_music = {
    'happy': 'Pop / Upbeat / Dance',
    'sad': 'Acoustic / Lofi / classical',
    'angry': 'Rock / Heavy Metal',
    'fear': 'Ambient / Calming',
    'surprise': 'Electronic / Synthwave',
    'disgust': 'Alternative / Indie',
    'neutral': 'Jazz / Lo-Fi Beats'
}

# Buffer for smoothing predictions
emotion_buffer = deque(maxlen=5)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.50


# 2. Main Detection Function

def run_detection():
    """
    Run real-time face emotion detection using webcam.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam initialized. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Sort faces by size (largest first)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

            # Process only the largest face
            (x, y, w, h) = faces[0]

            roi_gray = gray[y:y + h, x:x + w]

            try:
                # Preprocess face
                roi_resized = cv2.resize(roi_gray, (48, 48))
                roi_normalized = roi_resized / 255.0
                roi_reshaped = np.reshape(roi_normalized, (1, 48, 48, 1))

                # Prediction
                preds = emotion_model.predict(roi_reshaped, verbose=0)
                confidence = float(np.max(preds))
                emotion_idx = np.argmax(preds)

                # Apply confidence threshold
                if confidence >= CONFIDENCE_THRESHOLD:
                    predicted_emotion = emotion_labels[emotion_idx]
                else:
                    predicted_emotion = 'neutral'

                # Smooth prediction
                emotion_buffer.append(predicted_emotion)
                smooth_emotion = Counter(emotion_buffer).most_common(1)[0][0]

                # Draw bounding box
                color = (255, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Display emotion
                text = f"{smooth_emotion} ({int(confidence * 100)}%)"
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                # Display music recommendation
                music_text = f"Music: {emotion_to_music.get(smooth_emotion, 'Any')}"
                cv2.putText(frame, music_text, (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error processing face: {e}")

        # Show output window
        cv2.imshow('Face Emotion & Music Recommendation', frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()