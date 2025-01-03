import numpy as np
import os

# Function to normalize landmarks
def normalize_landmarks(landmarks):
    # Center the landmarks by subtracting the mean
    mean = np.mean(landmarks, axis=0)
    landmarks_centered = landmarks - mean
    
    # Scale the landmarks
    std_dev = np.std(landmarks_centered)
    if std_dev > 0:
        landmarks_normalized = landmarks_centered / std_dev
    else:
        landmarks_normalized = landmarks_centered

    return landmarks_normalized

# Function to augment data
def augment_landmarks(landmarks):
    augmented_samples = []

    # Original sample
    augmented_samples.append(landmarks)

    # Rotation
    for angle in [10, -10]:  # Small rotations
        rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle)), 0],
                                     [np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0],
                                     [0, 0, 1]])
        rotated_landmarks = np.dot(landmarks, rotation_matrix.T)
        augmented_samples.append(rotated_landmarks)

    # Adding noise
    noise = np.random.normal(0, 0.01, landmarks.shape)  # Adjust noise level as needed
    noisy_landmarks = landmarks + noise
    augmented_samples.append(noisy_landmarks)

    return augmented_samples

# Function to preprocess and save augmented data
def preprocess_gesture_data(gesture_dir):
    for gesture_file in os.listdir(gesture_dir):
        if gesture_file.endswith('.npy'):
            gesture_name = gesture_file.split('.')[0]
            landmarks = np.load(os.path.join(gesture_dir, gesture_file))

            # Normalize and augment landmarks
            normalized_samples = []
            for sample in landmarks:
                normalized_landmark = normalize_landmarks(sample)
                augmented_samples = augment_landmarks(normalized_landmark)
                normalized_samples.extend(augmented_samples)

            # Save the preprocessed data
            preprocessed_file = os.path.join(gesture_dir, f"preprocessed_{gesture_name}.npy")
            np.save(preprocessed_file, np.array(normalized_samples))
            print(f"Saved preprocessed samples for gesture '{gesture_name}' to {preprocessed_file}")

if __name__ == "__main__":
    preprocess_gesture_data('gestures')  # Adjust the directory if needed
