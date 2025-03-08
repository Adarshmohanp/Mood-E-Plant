# Mood-E-Plant

## Overview
A real-time emotion detection application that displays an interactive digital plant responding to user emotions. The plant's appearance and background music change based on the detected emotional state of the user through webcam input.

## Features
- Real-time facial emotion detection
- Interactive plant visualization
- Emotion-based background music
- Dynamic gradient backgrounds
- Seamless webcam integration
- Cross-platform compatibility

## Project Structure
```
Mood-E-Plant/
├── backend/
│   ├── assets/
│   │   ├── audio/
│   │   │   ├── happy_background.mp3
│   │   │   ├── sad_background.mp3
│   │   │   ├── angry_background.mp3
│   │   │   └── neutral_background.mp3
│   │   ├── models/
│   │   │   └── facemodel.keras
│   │   └── plant_images/
│   │       ├── happy_plant.png
│   │       ├── sad_plant.png
│   │       ├── angry_plant.png
│   │       └── neutral_plant.png
│   ├── src/
│   │   ├── audio_controller.py
│   │   ├── create_model.py
│   │   ├── convert_model.py
│   │   └── webcam_detector.py
│   └── requirements.txt
└── frontend/
    └── digital-plant-react/
        ├── public/
        └── src/
            ├── components/
            │   ├── EmotionDetector.tsx
            │   ├── Plant.tsx
            │   └── GradientBackground.tsx
            └── App.tsx
```

## Prerequisites
- Python 3.11+
- Node.js 18+
- npm 9+
- Webcam access
- Windows/Linux/MacOS

## Installation

### Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd frontend/digital-plant-react
npm install
```

## Running the Application

1. Start the backend server:
```bash
cd backend
.\venv\Scripts\activate  # On Windows
python src/webcam_detector.py
```

2. Start the frontend development server:
```bash
cd frontend/digital-plant-react
npm start
```

Access the application at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Technology Stack
### Backend
- Python 3.11
- TensorFlow (Deep Learning)
- OpenCV (Computer Vision)
- FastAPI (API Framework)
- Pygame (Audio)

### Frontend
- React with TypeScript
- Styled Components
- Framer Motion (Animations)
- Gradient Animations

## Emotion Detection Features
The application detects and responds to four primary emotions:
- Happy: Golden gradient background with upbeat music
- Sad: Blue gradient background with melancholic music
- Angry: Red gradient background with intense music
- Neutral: Green gradient background with calm music

## API Endpoints
- `GET /webcam-feed`: Returns detected emotion and plant image
- `GET /shutdown`: Gracefully shuts down the application

## Model Architecture
- Input: 48x48 grayscale images
- Convolutional Neural Network with:
  - 3 Convolutional blocks
  - Batch normalization
  - MaxPooling layers
  - Dropout for regularization
- Output: 4 emotion classes

## Requirements
```
numpy==1.24.3
tensorflow==2.13.0
opencv-python==4.8.1.78
fastapi==0.103.2
pygame==2.5.2
```

## Troubleshooting
Common issues and solutions:
1. Webcam access denied: Enable camera permissions
2. Audio not playing: Check audio device settings
3. Model detection issues: Ensure proper lighting and face positioning
4. Backend errors: Check console for specific error messages

## Known Limitations
- Works best in good lighting conditions
- Requires direct face visibility
- Limited to four basic emotions
- Single face detection at a time

## Future Improvements
- Multi-face detection support
- More emotion categories
- Improved model accuracy
- Additional plant animations
- Custom audio upload feature

## License
This project is licensed under the MIT License