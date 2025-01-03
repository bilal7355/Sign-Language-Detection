import cv2
import numpy as np
import mediapipe as mp
import os
import time

# Set up MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Directory to save gesture data
gesture_dir = 'gestures'
if not os.path.exists(gesture_dir):
    os.makedirs(gesture_dir)

# Function to capture gestures
def capture_gestures(gesture_name, num_samples=10):
    samples = []
    print(f"Capturing samples for gesture: {gesture_name}")

    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Allow camera to warm up

    while len(samples) < num_samples:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        # Process the frame for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmark positions to calculate bounding box
                h, w, _ = frame.shape
                landmark_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                x_min = min([pt[0] for pt in landmark_coords])
                y_min = min([pt[1] for pt in landmark_coords])
                x_max = max([pt[0] for pt in landmark_coords])
                y_max = max([pt[1] for pt in landmark_coords])

                # Draw bounding box around the hand
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Crop the hand region within the bounding box
                hand_region = frame[y_min:y_max, x_min:x_max]
                hand_region = cv2.resize(hand_region, (64, 64))  # Resize for consistency

                # Append landmarks and the cropped hand region for each sample
                landmarks = np.array([(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark])
                samples.append(landmarks)

                # Show the frame with bounding box and cropped hand region
                cv2.imshow('Hand Region', hand_region)
                cv2.putText(frame, f'Sample {len(samples)}/{num_samples}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Gesture Capture', frame)

                # Wait for a short duration before capturing the next sample
                time.sleep(1)  # Adjust delay as needed

        else:
            cv2.putText(frame, "No hand detected", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Gesture Capture', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the samples to a file
    samples_file = os.path.join(gesture_dir, f"{gesture_name}.npy")
    np.save(samples_file, np.array(samples))
    print(f"Saved {len(samples)} samples for gesture '{gesture_name}' to {samples_file}")

if __name__ == "__main__":
    while True:
        gesture_name = input("Enter the gesture name (or 'exit' to quit): ")
        if gesture_name.lower() == 'exit':
            break
        num_samples = int(input("Enter the number of samples to capture: "))
        capture_gestures(gesture_name, num_samples)
