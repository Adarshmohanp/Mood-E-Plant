import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from pygame import mixer  # For sound effects

# Load the emotion recognition model
model = tf.keras.models.load_model('facemodel.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load plant images
plant_images = {
    'Happy': Image.open('happy_plant.png'),
    'Neutral': Image.open('neutral_plant.png')  # Default image for non-happy emotions
}

# Load sound effects
mixer.init()
'''sound_effects = {
    'Happy': 'happy_sound.wav'
}
'''
# Face detection model
faceclass = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = None
running = False

# Function to update the plant image and sound
def update_plant(emotion):
    if emotion == 'Happy':  # Only update for happy emotion
        plant_image = plant_images.get('Happy')
        if plant_image:
            plant_image = plant_image.resize((300, 300), Image.Resampling.LANCZOS)
            plant_photo = ImageTk.PhotoImage(plant_image)
            plant_label.config(image=plant_photo)
            plant_label.image = plant_photo

            '''if emotion in sound_effects:
                mixer.music.load(sound_effects[emotion])
                mixer.music.play()'''
    else:
        # Display default image for non-happy emotions
        plant_image = plant_images.get('Neutral')
        if plant_image:
            plant_image = plant_image.resize((300, 300), Image.Resampling.LANCZOS)
            plant_photo = ImageTk.PhotoImage(plant_image)
            plant_label.config(image=plant_photo)
            plant_label.image = plant_photo

# Function to detect faces and predict emotions
def detectbox(vid):
    grayimage = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = faceclass.detectMultiScale(grayimage, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        face = grayimage[y:y+h, x:x+w]
        if face.size > 0:  # Ensure the face region is valid
            # Add padding to the face region
            face = cv2.copyMakeBorder(face, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255
            face = tf.keras.preprocessing.image.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            prediction = model.predict(face)
            emotion = emotion_labels[np.argmax(prediction)]
            cv2.putText(vid, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            update_plant(emotion)  # Update the plant based on emotion
    return vid

# Function to start webcam
def start_webcam():
    global cam, running
    if not running:
        running = True
        cam = cv2.VideoCapture(0)
        show_frame()

# Function to stop webcam
def stop_webcam():
    global cam, running
    running = False
    if cam is not None:
        cam.release()
        cam = None
    webcam_label.config(image='')

# Function to continuously update video feed
def show_frame():
    global cam, running
    if running and cam is not None:
        ret, frame = cam.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = detectbox(frame)  # Detect emotions and update plant
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 450))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            webcam_label.imgtk = imgtk
            webcam_label.configure(image=imgtk)
        if running:
            root.after(10, show_frame)  # Asynchronous update

# Main Tkinter window
root = tk.Tk()
root.title("Moody Plant Lite - Happy Test")
root.geometry("800x600")
root.configure(bg="#1e1e1e")

# Plant display
plant_label = tk.Label(root, bg="#1e1e1e")
plant_label.pack(pady=20)

# Webcam feed
webcam_label = tk.Label(root, bg="#1e1e1e")
webcam_label.pack()

# Buttons
btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=10)

btn_style1 = {"font": ("Arial", 14, "bold"), "fg": "white", "bg": "lawngreen", "width": 20, "bd": 0, "relief": "flat"}
btn_style2 = {"font": ("Arial", 14, "bold"), "fg": "white", "bg": "red", "width": 20, "bd": 0, "relief": "flat"}

btn_start = tk.Button(btn_frame, text="Start Webcam", command=start_webcam, **btn_style1)
btn_start.pack(side=tk.LEFT, padx=10, pady=5)

btn_stop = tk.Button(btn_frame, text="Stop Webcam", command=stop_webcam, **btn_style2)
btn_stop.pack(side=tk.RIGHT, padx=10, pady=5)

# Start the app
root.mainloop()