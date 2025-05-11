from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "asl_gesture_recognition_model.h5")
model = tf.keras.models.load_model(model_path)

# Define the labels
labels = ['hello', 'yes', 'no', 'i love you', 'thank you']

# Image size to match trained model
img_height = 64
img_width = 64

# HSV range for skin detection
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# Define a threshold for minimum confidence
CONFIDENCE_THRESHOLD = 40.0

def process_frame(frame):
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Find contours to detect hand region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_detected = False
    prediction_text = "No hand detected"
    
    if contours:
        # Get the largest contour (assumed to be the hand)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500:
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
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Preprocess the hand region
            hand_resized = cv2.resize(hand_region, (img_width, img_height))
            img = hand_resized / 255.0
            img = np.expand_dims(img, axis=0)

            # Perform inference
            predictions = model.predict(img, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            predicted_label = labels[predicted_idx]
            confidence = predictions[0][predicted_idx] * 100

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_text = f"{predicted_label} ({confidence:.1f}%)"
                cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                prediction_text = "Low confidence"
                cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if not hand_detected:
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return frame

def generate_frames():
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print(f"Successfully opened camera {camera_index}")
                break
        except Exception as e:
            print(f"Failed to open camera {camera_index}: {e}")
            continue
    else:
        print("No camera could be opened")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
        else:
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 