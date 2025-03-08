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
    """Enhanced preprocessing with better face normalization"""
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

def get_emotion_prediction(face_img, confidence_threshold=0.4):
    """Get emotion prediction with confidence threshold and ensemble"""
    try:
        # Get base prediction
        base_pred = model.predict(face_img, verbose=0)[0]
        
        # Get predictions with slight rotations for robustness
        predictions = [base_pred]
        angles = [-5, 5]
        
        # Get image from tensor
        img = face_img[0, :, :, 0]  # Remove batch and channel dimensions
        
        for angle in angles:
            # Rotate using OpenCV instead of tensorflow/scipy
            height, width = img.shape[:2]
            center = (width/2, height/2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
            
            # Add batch and channel dimensions back
            rotated = np.expand_dims(rotated, axis=-1)  # Add channel dimension
            rotated = np.expand_dims(rotated, axis=0)   # Add batch dimension
            
            # Get prediction for rotated image
            pred = model.predict(rotated, verbose=0)[0]
            predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        
        # Get top 2 predictions
        top_2_idx = np.argsort(avg_pred)[-2:]
        emotion_idx = top_2_idx[-1]
        confidence = float(avg_pred[emotion_idx])
        
        # If confidence is too low, consider second prediction
        if confidence < confidence_threshold:
            second_idx = top_2_idx[-2]
            second_confidence = float(avg_pred[second_idx])
            if second_confidence > confidence * 0.8:
                emotion_idx = second_idx
                confidence = second_confidence
        
        return emotion_idx, confidence
        
    except Exception as e:
        print(f"Error in get_emotion_prediction: {e}")
        return None, 0.0

# Initialize audio controller
try:
    audio_controller = AudioController()
    print("Audio controller initialized successfully")
except Exception as e:
    print(f"Error initializing audio controller: {e}")
    audio_controller = None

# Update the audio handling in get_webcam_feed function
@app.get("/webcam-feed")
async def get_webcam_feed():
    try:
        ret, frame = cap.read()
        if not ret:
            return {"error": "Failed to capture frame"}
        
        # Improve face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            maxSize=(300, 300)
        )
        
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
                
                # Process face
                processed_face = preprocess_face(face_roi)
                if processed_face is None:
                    continue
                
                # Get emotion prediction
                emotion_idx, confidence = get_emotion_prediction(
                    processed_face,
                    confidence_threshold=0.4
                )
                
                if emotion_idx is not None:
                    emotion = emotion_labels[emotion_idx]
                    
                    # Get plant image
                    plant_image_path = os.path.join(
                        os.path.dirname(__file__), 
                        f'../assets/plant_images/{emotion.lower()}_plant.png'
                    )
                    
                    plant_image = None
                    if os.path.exists(plant_image_path):
                        with open(plant_image_path, 'rb') as img_file:
                            plant_image = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    results.append({
                        "emotion": emotion,
                        "confidence": confidence,
                        "plant_image": plant_image
                    })
                    
                    # Update music if confidence is high enough
                    if confidence > 0.4:  # Lower the threshold to 0.4
                        audio_controller.play_emotion_music(emotion)
                        print(f"Detected emotion: {emotion} with confidence: {confidence}")
            
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
        
        if not results:
            audio_controller.stop_music()
        
        return {"results": results}
        
    except Exception as e:
        print(f"Error in webcam-feed: {e}")
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