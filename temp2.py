import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import sys
from contextlib import contextmanager

# Suppress Pygame welcome message
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

with suppress_stdout():
    from pygame import mixer  # For sound effects
    mixer.init()

# Load the emotion recognition model
model = tf.keras.models.load_model('facemodel.keras')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load plant images
plant_images = {
    'Happy': Image.open('happy_plant.png'),
    'Neutral': Image.open('neutral_plant.png'),
    'Sad': Image.open('sad_plant.png'),
    'Angry': Image.open('angry_plant.png'),
    'Fear': Image.open('fear_plant.png'),  # Add a plant image for Fear
    'Surprise': Image.open('surprise_plant.png')  # Add a plant image for Surprise
}

# Sassy messages for each emotion
sassy_messages = {
    'Happy': "You’re glowing today! 🌟",
    'Sad': "Even plants cry sometimes. 💧",
    'Angry': "Chill out before I wilt! 🔥",
    'Fear': "Don’t be scared, I’m here! 👻",
    'Surprise': "Wow, you surprised me! 😲",
    'Neutral': "Just another day in the pot. 🪴",
    
}

# Face detection model
faceclass = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = None
running = False

# Function to update the plant image and sound
def update_plant(emotion):
    plant_image = plant_images.get(emotion)
    if plant_image:
        # Resize the plant image to fit the window
        plant_image = plant_image.resize((400, 400), Image.Resampling.LANCZOS)
        plant_photo = ImageTk.PhotoImage(plant_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo

        # Update the sassy message
        message = sassy_messages.get(emotion, "I’m feeling something... 🤔")
        message_label.config(text=message)

        # Start animation based on emotion
        if emotion == 'Happy':
            animate_sway(plant_image)
        elif emotion == 'Sad':
            animate_droop(plant_image)
        elif emotion == 'Angry':
            animate_shake(plant_image)
        elif emotion == 'Fear':
            animate_fear(plant_image)  # Animation for Fear
        elif emotion == 'Surprise':
            animate_surprise(plant_image)  # Animation for Surprise
        elif emotion == 'Neutral':
            # No animation for neutral
            pass

# Function to animate swaying (for happy emotion)
def animate_sway(plant_image):
    for angle in range(-10, 11, 5):  # Rotate from -10 to 10 degrees
        rotated_image = plant_image.rotate(angle, expand=True)
        plant_photo = ImageTk.PhotoImage(rotated_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo
        root.update()  # Update the GUI
        root.after(100)  # Delay for smooth animation

# Function to animate drooping (for sad emotion)
def animate_droop(plant_image):
    for angle in range(0, 20, 5):  # Rotate from 0 to 20 degrees downward
        rotated_image = plant_image.rotate(angle, expand=True)
        plant_photo = ImageTk.PhotoImage(rotated_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo
        root.update()  # Update the GUI
        root.after(100)  # Delay for smooth animation

# Function to animate shaking (for angry emotion)
def animate_shake(plant_image):
    for _ in range(5):  # Shake 5 times
        for angle in range(-10, 11, 5):  # Rotate from -10 to 10 degrees
            rotated_image = plant_image.rotate(angle, expand=True)
            plant_photo = ImageTk.PhotoImage(rotated_image)
            plant_label.config(image=plant_photo)
            plant_label.image = plant_photo
            root.update()  # Update the GUI
            root.after(50)  # Delay for fast shaking

# Function to animate fear (for fear emotion)
def animate_fear(plant_image):
    def shrink(scale):
        if not running:  # Stop animation if webcam is stopped
            return
        # Resize the plant image
        new_size = (int(400 * scale), int(400 * scale))
        resized_image = plant_image.resize(new_size, Image.Resampling.LANCZOS)
        plant_photo = ImageTk.PhotoImage(resized_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo
        # Schedule the next frame
        if scale > 0.8:  # Shrink to 80% of the original size
            root.after(100, shrink, scale - 0.05)
        else:
            root.after(100, grow, scale + 0.05)

    def grow(scale):
        if not running:  # Stop animation if webcam is stopped
            return
        # Resize the plant image
        new_size = (int(400 * scale), int(400 * scale))
        resized_image = plant_image.resize(new_size, Image.Resampling.LANCZOS)
        plant_photo = ImageTk.PhotoImage(resized_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo
        # Schedule the next frame
        if scale < 1.0:  # Grow back to the original size
            root.after(100, grow, scale + 0.05)
        else:
            root.after(100, tremble, 0)

    def tremble(count):
        if not running:  # Stop animation if webcam is stopped
            return
        # Slightly shake the plant
        angle = 5 if count % 2 == 0 else -5
        rotated_image = plant_image.rotate(angle, expand=True)
        plant_photo = ImageTk.PhotoImage(rotated_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo
        # Schedule the next frame
        if count < 10:  # Tremble 10 times
            root.after(50, tremble, count + 1)

    shrink(1.0)  # Start the shrink animation

# Function to animate surprise (for surprise emotion)
def animate_surprise(plant_image):
    def jump(offset):
        if not running:  # Stop animation if webcam is stopped
            return
        # Move the plant up and down (only adjust the Y position)
        plant_label.place(x=100, y=50 + offset)  # Explicitly set the X position
        # Schedule the next frame
        if offset > -20:  # Move up
            root.after(50, jump, offset - 10)
        elif offset < 20:  # Move down
            root.after(50, jump, offset + 10)
        else:
            plant_label.place(x=100, y=50)  # Reset position (explicitly set X and Y)
            root.after(50, expand, 1.0)

    def expand(scale):
        if not running:  # Stop animation if webcam is stopped
            return
        # Resize the plant image
        new_size = (int(400 * scale), int(400 * scale))
        resized_image = plant_image.resize(new_size, Image.Resampling.LANCZOS)
        plant_photo = ImageTk.PhotoImage(resized_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo
        # Schedule the next frame
        if scale < 1.2:  # Expand to 120% of the original size
            root.after(50, expand, scale + 0.05)
        else:
            root.after(50, shrink_back, scale - 0.05)

    def shrink_back(scale):
        if not running:  # Stop animation if webcam is stopped
            return
        # Resize the plant image
        new_size = (int(400 * scale), int(400 * scale))
        resized_image = plant_image.resize(new_size, Image.Resampling.LANCZOS)
        plant_photo = ImageTk.PhotoImage(resized_image)
        plant_label.config(image=plant_photo)
        plant_label.image = plant_photo
        # Schedule the next frame
        if scale > 1.0:  # Shrink back to the original size
            root.after(50, shrink_back, scale - 0.05)

    jump(0)  # Start the jump animation

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
            update_plant(emotion)  # Update the plant based on emotion
    return vid

# Function to start webcam
def start_webcam():
    global cam, running
    if not running:
        running = True
        cam = cv2.VideoCapture(0)
        btn_start.pack_forget()  # Remove the Start button from the GUI
        show_frame()

# Function to continuously update video feed (without displaying it)
def show_frame():
    global cam, running
    if running and cam is not None:
        ret, frame = cam.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = detectbox(frame)  # Detect emotions and update plant
        if running:
            root.after(10, show_frame)  # Asynchronous update

# Function to handle application closing
def on_closing():
    global cam, running
    running = False
    if cam is not None:
        cam.release()
    root.destroy()

# Main Tkinter window
root = tk.Tk()
root.title("Moody Plant Lite")
root.geometry("600x700")  # Larger window for better visual appeal
root.configure(bg="#1e1e1e")

# Plant display
plant_label = tk.Label(root, bg="#1e1e1e")
plant_label.pack(pady=50)  # Add more padding for better spacing

# Message display (fixed position using place)
message_label = tk.Label(root, text="How are you feeling today? 🌱", font=("Arial", 16), fg="white", bg="#1e1e1e")
message_label.place(x=50, y=20)  # Fixed position at the top

# Buttons
btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=20)

btn_style1 = {"font": ("Arial", 14, "bold"), "fg": "white", "bg": "lawngreen", "width": 20, "bd": 0, "relief": "flat"}

# Start button
btn_start = tk.Button(btn_frame, text="Start Webcam", command=start_webcam, **btn_style1)
btn_start.pack(side=tk.LEFT, padx=10, pady=5)

# Handle window closing event
root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the app
root.mainloop()
