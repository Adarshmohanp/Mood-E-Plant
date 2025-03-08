import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

def check_dependencies():
    required_packages = {
        'numpy': 'numpy',
        'opencv-python': 'cv2',
        'tensorflow': 'tensorflow',
        'fastapi': 'fastapi',
        'Pillow': 'PIL'
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", ", ".join(missing_packages))
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

check_dependencies()

# Import required packages
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import base64
from PIL import Image

# Add imports
from audio_controller import AudioController
import pygame

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize variables
model = None
face_cascade = None
cap = None

try:
    model_path = os.path.join(os.path.dirname(__file__), '../assets/models/facemodel.keras')
    if os.path.exists(model_path):
        try:
            # Load model with custom_objects to handle BatchNormalization layers
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={
                'BatchNormalization': tf.keras.layers.BatchNormalization
            })
            print("Model loaded successfully")
        except Exception as e:
            print(f"Could not load model: {e}")
            print("Creating new model...")
            from create_model import create_emotion_model
            model = create_emotion_model()
            print("New model created successfully")
    else:
        print("No model found. Creating new model...")
        from create_model import create_emotion_model
        model = create_emotion_model()
        print("New model created successfully")

    # Compile the model with same parameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Initialize camera with specific resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        raise Exception("Could not open video capture device")

except Exception as e:
    print(f"Error during initialization: {e}")
    if cap is not None:
        cap.release()
    sys.exit(1)

# Update emotion labels to only include the ones we want
emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad']

def preprocess_face(face_img):
    """Enhanced preprocessing for improved emotion detection"""
    try:
        # Ensure grayscale
        if len(face_img.shape) > 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Add padding to ensure face features are captured
        h, w = face_img.shape
        padding = int(max(h, w) * 0.2)
        face_img = cv2.copyMakeBorder(
            face_img, 
            padding, padding, padding, padding,
            cv2.BORDER_REPLICATE
        )
        
        # Apply histogram equalization
        face_img = cv2.equalizeHist(face_img)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        face_img = clahe.apply(face_img)
        
        # Gaussian blur to reduce noise
        face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
        
        # Resize to model input size
        face_img = cv2.resize(face_img, (48, 48))
        
        # Normalize to [-1, 1] range
        face_img = face_img.astype('float32')
        face_img = (face_img - 127.5) / 127.5
        
        # Add dimensions for model
        face_img = np.expand_dims(face_img, axis=-1)
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    except Exception as e:
        print(f"Error in preprocess_face: {e}")
        return None

# Initialize audio controller
audio_controller = AudioController()

# Update the audio handling in get_webcam_feed function
@app.get("/webcam-feed")
async def get_webcam_feed():
    try:
        ret, frame = cap.read()
        if not ret:
            audio_controller.stop_music()  # Stop music if frame capture fails
            return {"error": "Failed to capture frame"}
        
        # Process frame with improved face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # More precise detection
            minNeighbors=6,
            minSize=(60, 60),
            maxSize=(300, 300)
        )
        
        if len(faces) == 0:
            audio_controller.stop_music()  # Stop music if no faces detected
            return {"results": []}
        
        results = []
        for (x, y, w, h) in faces:
            try:
                # Extract face with margin
                margin = int(max(w, h) * 0.2)
                y1 = max(0, y - margin)
                y2 = min(frame.shape[0], y + h + margin)
                x1 = max(0, x - margin)
                x2 = min(frame.shape[1], x + w + margin)
                
                face_roi = gray[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                
                # Preprocess face
                processed_face = preprocess_face(face_roi)
                if processed_face is None:
                    continue
                
                # Get prediction with temperature scaling
                prediction = model.predict(processed_face, verbose=0)[0]
                temperature = 0.7  # Adjust prediction sharpness
                prediction = np.exp(np.log(prediction) / temperature)
                prediction = prediction / np.sum(prediction)
                
                # Get top 2 predictions
                top_2_idx = np.argsort(prediction)[-2:]
                emotion_idx = top_2_idx[-1]
                confidence = float(prediction[emotion_idx])
                
                # Emotion-specific thresholds
                thresholds = {
                    'Happy': 0.4,
                    'Sad': 0.4,
                    'Angry': 0.45,
                    'Neutral': 0.35
                }
                
                emotion = emotion_labels[emotion_idx]
                
                # Check confidence threshold
                if confidence < thresholds.get(emotion, 0.4):
                    # Try second best prediction
                    second_idx = top_2_idx[-2]
                    second_confidence = float(prediction[second_idx])
                    if second_confidence > confidence * 0.8:
                        emotion_idx = second_idx
                        emotion = emotion_labels[emotion_idx]
                        confidence = second_confidence
                
                # Play corresponding music
                audio_controller.play_emotion_music(emotion)
                
                # Get plant image
                plant_image_path = os.path.join(
                    os.path.dirname(__file__), 
                    f'../assets/plant_images/{emotion.lower()}_plant.png'
                )
                
                if os.path.exists(plant_image_path):
                    with open(plant_image_path, 'rb') as img_file:
                        plant_image = base64.b64encode(img_file.read()).decode('utf-8')
                else:
                    plant_image = None
                
                results.append({
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "plant_image": plant_image
                })
            
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        return {
            "results": results
        }
    except Exception as e:
        print(f"Error in webcam-feed: {e}")
        audio_controller.stop_music()  # Stop music on error
        return {"error": str(e)}

# Update shutdown event
@app.on_event("shutdown")
def shutdown_event():
    try:
        if cap is not None:
            cap.release()
        audio_controller.stop_music()
        pygame.mixer.quit()
    except Exception as e:
        print(f"Error during shutdown: {e}")

if __name__ == "__main__":
    import uvicorn
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        shutdown_event()