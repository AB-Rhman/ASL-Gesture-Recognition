import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Define the model
img_height = 64
img_width = 64
num_classes = 5

# Create base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

# Create the full model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the model
model_path = os.path.join("models", "asl_gesture_recognition_model.h5")
model.save(model_path)
print(f"Model saved to {model_path}") 