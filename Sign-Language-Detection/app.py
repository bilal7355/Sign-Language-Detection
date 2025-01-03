import numpy as np
import cv2
import mediapipe as mp
import os
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3
from flask import Flask, render_template, Response, jsonify
import threading
import queue

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize Text-to-Speech
engine = pyttsx3.init()
speech_queue = queue.Queue()
running = True
is_speaking = False  # Flag to check if the engine is busy
speech_lock = threading.Lock()  # Lock for speech engine access

def speech_thread():
    global is_speaking
    while running:
        try:
            gesture = speech_queue.get(timeout=1)  # Wait for a gesture to be added to the queue
            with speech_lock:  # Acquire the lock before accessing the speech engine
                if not is_speaking:  # Check if the engine is already speaking
                    is_speaking = True
                    engine.say(gesture)
                    engine.runAndWait()
                    is_speaking = False  # Reset the flag after speaking
        except queue.Empty:
            continue

# Load preprocessed data
def load_gesture_data(gesture_dir):
    X = []
    y = []
    for gesture_file in os.listdir(gesture_dir):
        if gesture_file.startswith('preprocessed_'):
            gesture_name = gesture_file.split('_')[1].split('.')[0]
            landmarks = np.load(os.path.join(gesture_dir, gesture_file))
            for landmark in landmarks:
                X.append(landmark.flatten())  # Flatten each sample
                y.append(gesture_name)
    return np.array(X), np.array(y)

# Train k-NN model
gesture_dir = 'gestures'
X, y = load_gesture_data(gesture_dir)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Function for normalization
def normalize_landmarks(landmarks):
    mean = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - mean
    std_dev = np.std(landmarks_centered)
    landmarks_normalized = landmarks_centered / std_dev if std_dev > 0 else landmarks_centered
    return landmarks_normalized.flatten()  # Ensure it’s 1D

# Function for gesture recognition
def recognize_gesture(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark])
            landmarks_normalized = normalize_landmarks(landmarks).reshape(1, -1)  # Flatten to 2D for prediction

            # Predict gesture
            predicted_gesture = knn.predict(landmarks_normalized)[0]
            return predicted_gesture

    return None

# Video capture generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    last_gesture = None  # Variable to track the last recognized gesture
    while True:
        success, frame = cap.read()
        if not success:
            break

        recognized_gesture = recognize_gesture(frame)

        if recognized_gesture:
            cv2.putText(frame, f"Gesture: {recognized_gesture}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add the recognized gesture to the speech queue only if it's different from the last one
            if recognized_gesture != last_gesture:
                speech_queue.put(recognized_gesture)  # Put gesture into the queue
                last_gesture = recognized_gesture  # Update the last recognized gesture
        else:
            last_gesture = None

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognized_gesture')
def recognized_gesture():
    # Return the last recognized gesture
    gesture = speech_queue.queue[-1] if not speech_queue.empty() else None
    return jsonify({'gesture': gesture})

if __name__ == "__main__":
    threading.Thread(target=speech_thread, daemon=True).start()  # Start the speech thread
    try:
        app.run(debug=True)
    except KeyboardInterrupt:
        running = False  # Stop the speech thread on exit
