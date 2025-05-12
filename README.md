# ASL Gesture Recognition

A real-time American Sign Language (ASL) gesture recognition system that can recognize five common signs using computer vision and deep learning.

## Features

- Real-time hand gesture recognition
- Supports 5 ASL signs:
  - Hello
  - Yes
  - No
  - I love you
  - Thank you
- Web-based interface
- Real-time video feed with gesture detection
- Confidence score display
- Docker support for easy deployment

## Requirements

- Python 3.9 or higher
- Webcam
- Required Python packages (listed in requirements.txt):
  - Flask
  - OpenCV
  - NumPy
  - TensorFlow 2.13.1
- Docker and Docker Compose (for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd asl-gesture-recognition
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Docker Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd asl-gesture-recognition
```

2. Build and run using Docker Compose:
```bash
# Clean up any existing containers and images
docker-compose down
docker system prune -f

# Build and start the application
docker-compose up --build
```

## Usage

### Local Usage

1. Run the application:
```bash
python app.py
```

2. Open your web browser and go to:
```
http://localhost:5000
```

### Docker Usage

1. Start the application:
```bash
docker-compose up
```

2. Open your web browser and go to:
```
http://localhost:5000
```

3. To stop the application:
```bash
docker-compose down
```

## How to Use

1. Make sure you have good lighting
2. Position your hand clearly in front of the camera
3. The system will detect your hand and show a green box around it
4. The recognized gesture will be displayed at the top of the video
5. A confidence score will be shown with each prediction

## Project Structure

```
asl-gesture-recognition/
├── app.py              # Main Flask application
├── models/             # Contains the trained model
│   └── asl_gesture_recognition_model.h5
├── templates/          # HTML templates
│   └── index.html
├── requirements.txt    # Python dependencies
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
└── README.md          # This file
```

## Docker Configuration

The project includes Docker support for easy deployment:

- `Dockerfile`: Configures the Python environment and dependencies
- `docker-compose.yml`: Orchestrates the application and webcam access

### Docker Features

- Uses Python 3.9 slim image for optimal compatibility
- Includes all necessary system dependencies (OpenCV, TensorFlow)
- Enables webcam access through device mapping
- Mounts the current directory for live code changes
- Exposes port 5000 for web access
- Includes TensorFlow model verification step
- Optimized layer caching for faster builds

### Docker Commands

- Build and start: `docker-compose up --build`
- Start in background: `docker-compose up -d`
- Stop: `docker-compose down`
- View logs: `docker-compose logs -f`
- Rebuild: `docker-compose up --build --force-recreate`
- Clean up: `docker system prune -f`

## Troubleshooting

### Common Issues

1. Webcam Access:
   - Ensure Docker has permission to access your webcam
   - On Linux, you might need to add your user to the video group
   - On Windows, ensure the webcam is not being used by another application

2. Build Issues:
   - If you encounter build errors, try cleaning up Docker:
     ```bash
     docker-compose down
     docker system prune -f
     docker-compose up --build
     ```

3. Memory Issues:
   - If you encounter memory errors during build, try:
     ```bash
     docker-compose build --no-cache
     ```

## Notes

- The system works best with good lighting conditions
- Keep your hand within the camera frame
- The confidence threshold is set to 40%
- Press 'q' to quit the application
- For Docker users, ensure Docker has permission to access your webcam
- The application uses TensorFlow 2.13.1 for optimal compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Flask, OpenCV, and TensorFlow
- Uses MobileNetV2 as the base model for feature extraction
- Containerized with Docker for easy deployment 