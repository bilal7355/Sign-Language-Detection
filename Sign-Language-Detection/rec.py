import numpy as np
import cv2
import mediapipe as mp
import os
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize Text-to-Speech
engine = pyttsx3.init()

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
    if std_dev > 0:
        landmarks_normalized = landmarks_centered / std_dev
    else:
        landmarks_normalized = landmarks_centered
    return landmarks_normalized.flatten()  # Ensure it’s 1D

# Function for gesture recognition
def recognize_gesture(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmarks
            landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark])
            landmarks_normalized = normalize_landmarks(landmarks).reshape(1, -1)  # Flatten to 2D for prediction

            # Predict gesture
            predicted_gesture = knn.predict(landmarks_normalized)[0]
            return predicted_gesture

    return None

# Main function to run gesture recognition
def main():
    cap = cv2.VideoCapture(0)
    last_gesture = None  # Variable to track the last recognized gesture
    print("Gesture recognition started. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        recognized_gesture = recognize_gesture(frame)

        if recognized_gesture:
            cv2.putText(frame, f"Gesture: {recognized_gesture}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Speak the recognized gesture only if it's different from the last one
            if recognized_gesture != last_gesture:
                engine.say(recognized_gesture)
                engine.runAndWait()
                last_gesture = recognized_gesture  # Update the last recognized gesture

        else:
            # Reset last_gesture when no gesture is detected
            last_gesture = None

        cv2.imshow('Gesture Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
