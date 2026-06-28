import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf

# Emotion labels
emotion_labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# Haar Cascade Face Detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# -------------------------------
# Load Emotion Model
# -------------------------------
def load_emotion_model(path):
    return load_model(path, compile=False)

# -------------------------------
# Detect Faces
# -------------------------------
def detect_faces(frame):

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    return faces, gray

# -------------------------------
# Preprocess Face
# -------------------------------
def preprocess_face(face):

    # Convert to grayscale if needed
    if len(face.shape) == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    face = cv2.equalizeHist(face)

    # Resize to model input size
    face = cv2.resize(face, (48, 48))

    # Normalize
    face = face / 255.0

    # Reshape for CNN
    face = np.reshape(face, (1, 48, 48, 1))

    return face