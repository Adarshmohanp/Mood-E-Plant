# Mood-E-Plant

## Overview
A real-time emotion detection application that displays an interactive digital plant responding to user emotions. The plant's appearance and animations change based on the detected emotional state of the user through webcam input.

## Features
- Real-time facial emotion detection
- Interactive plant visualization
- Dynamic emotion-based animations
- Seamless webcam integration
- Cross-platform compatibility

## Project Structure
```
Mood-E-Plant/
├── backend/
│   ├── assets/
│   │   ├── pot.png
│   │   ├── stem.png
│   │   ├── leaves.png
│   │   ├── flower.png
│   │   ├── petal.png
│   │   └── background.png
│   ├── requirements.txt
│   └── moody_plant.py
└── frontend/
    └── digital-plant-react/
        ├── public/
        │   └── index.html
        └── src/
            ├── components/
            │   ├── EmotionDisplay.tsx
            │   ├── Plant.tsx
            │   └── WebcamFeed.tsx
            ├── services/
            │   └── emotionService.ts
            └── App.tsx
```

## Prerequisites
- Python 3.8+
- Node.js 14+
- npm 6+
- Webcam access
- Windows/Linux/MacOS

## Installation

### Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate  # On Windows
source venv/bin/activate # On Linux/MacOS
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
python moody_plant.py
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
- Python 3.8+
- OpenCV (Computer Vision)
- DeepFace (Emotion Detection)
- FastAPI (API Framework)
- Pygame (Graphics)

### Frontend
- React 18
- TypeScript
- Framer Motion (Animations)
- Axios (HTTP Client)

## Emotion Detection Features
The application detects and responds to the following emotions:
- Happy: Plant blooms and glows
- Sad: Plant droops slightly
- Angry: Plant shakes gently
- Neutral: Plant maintains steady state
- Surprised: Plant displays rapid movement
- Fear: Plant shows protective posture
- Disgust: Plant exhibits recoiling animation

## API Endpoints
- `GET /emotion`: Returns current emotion and plant state
- `GET /health`: API health check

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development
### Backend Development
```bash
cd backend
python -m pytest tests/
```

### Frontend Development
```bash
cd frontend/digital-plant-react
npm test
npm run lint
```

## Troubleshooting
Common issues and solutions:
1. Webcam access denied: Enable camera permissions
2. Backend connection failed: Ensure Python server is running
3. Frontend not updating: Check browser console for errors

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenCV team for computer vision capabilities
- DeepFace for emotion detection
- React team for the frontend framework
- All contributors who have helped shape this project