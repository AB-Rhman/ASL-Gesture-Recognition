import cv2
import numpy as np
import tensorflow as tf
import time
import os

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)

# Load the trained model
model_path = os.path.join(project_dir, "models", "asl_gesture_recognition_model.h5")
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
except Exception as e:
    print(f"Error: Failed to load model from '{model_path}'. Reason: {e}")
    print("Please make sure the model file exists in the 'models' directory")
    exit()

# Define the labels for the ASL signs (aligned with training script)
labels = ['hello', 'yes', 'no', 'i love you', 'thank you']

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam not accessible!")
    exit()
print("Webcam is accessible. Starting real-time inference...")

# Image size to match trained model
img_height = 64
img_width = 64

# HSV range for skin detection
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Define a threshold for minimum confidence
CONFIDENCE_THRESHOLD = 40.0  # Lowered for MobileNetV2

# For FPS calculation
prev_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # Preprocess the frame
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours to detect hand region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_detected = False
    if contours:
        # Get the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500:  # Adjusted threshold
            hand_detected = True
            # Get bounding box around the hand
            x, y, w, h = cv2.boundingRect(max_contour)
            # Extract hand region with some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            hand_region = frame[y:y+h, x:x+w]
            # Draw bounding box on the frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Preprocess the hand region
            hand_resized = cv2.resize(hand_region, (img_width, img_height))
            img = hand_resized / 255.0  # Normalize to [0, 1]
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Perform inference
            predictions = model.predict(img, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_label = labels[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100  # Confidence in percentage

            # Display prediction if above threshold
            if confidence > CONFIDENCE_THRESHOLD:
                display_text = f"{predicted_label} ({confidence:.1f}%)"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Low confidence", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if not hand_detected:
        cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    try:
        cv2.imshow('Real-Time Gesture Recognition', frame)
    except cv2.error as e:
        print(f"Error: Failed to display frame with cv2.imshow. Reason: {e}")
        break

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()