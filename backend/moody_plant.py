import cv2
try:
    from deepface import DeepFace
except ImportError as e:
    print(f"Error importing DeepFace: {e}")
    print("Please ensure you have the correct versions of tensorflow and deepface installed")
    sys.exit(1)

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import io
import pygame
import sys
import os
import random
import time

# Define assets path
ASSETS_PATH = os.path.join(os.path.dirname(__file__), "assets")

# Initialize Pygame
pygame.init()

# Set up the screen
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Moody Plant")

# Load images with proper path handling
try:
    pot_img = pygame.image.load(os.path.join(ASSETS_PATH, "pot.png"))
    stem_img = pygame.image.load(os.path.join(ASSETS_PATH, "stem.png"))
    leaves_img = pygame.image.load(os.path.join(ASSETS_PATH, "leaves.png"))
    flower_img = pygame.image.load(os.path.join(ASSETS_PATH, "flower.png"))
    petal_img = pygame.image.load(os.path.join(ASSETS_PATH, "petal.png"))
    background_img = pygame.image.load(os.path.join(ASSETS_PATH, "background.png"))
except FileNotFoundError as e:
    print(f"Error loading images: {e}")
    print("Please ensure all required images are in the assets folder:")
    print("- pot.png\n- stem.png\n- leaves.png\n- flower.png\n- petal.png\n- background.png")
    sys.exit(1)

# Resize images (if needed)
pot_img = pygame.transform.scale(pot_img, (100, 100))
stem_img = pygame.transform.scale(stem_img, (20, 200))
leaves_img = pygame.transform.scale(leaves_img, (100, 100))
flower_img = pygame.transform.scale(flower_img, (50, 50))
petal_img = pygame.transform.scale(petal_img, (20, 20))  # Resize petal image

# Animation variables
sway_angle = 0
sway_direction = 1
shake_intensity = 0
falling_leaves = []
falling_petals = []
blooming_flowers = []

# Function to detect emotion
def detect_emotion():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return "neutral"  # Default emotion if webcam fails

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            return "neutral"

        cv2.imwrite("temp.jpg", frame)
        try:
            result = DeepFace.analyze("temp.jpg", actions=["emotion"], enforce_detection=False)
            emotion = result[0]["dominant_emotion"]
            print(f"Detected emotion: {emotion}")
            return emotion
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return "neutral"
        finally:
            if os.path.exists("temp.jpg"):
                try:
                    os.remove("temp.jpg")
                except:
                    pass
    except Exception as e:
        print(f"Unexpected error in detect_emotion: {e}")
        return "neutral"
    finally:
        cap.release()

# Function to draw the plant
def draw_plant(emotion):
    global sway_angle, sway_direction, shake_intensity, falling_leaves, falling_petals, blooming_flowers

    # Draw background
    screen.blit(background_img, (0, 0))

    # Draw pot
    screen.blit(pot_img, (350, 400))

    # Sway the stem and leaves
    sway_angle += 0.05 * sway_direction
    if abs(sway_angle) > 5:
        sway_direction *= -1

    # Rotate stem and leaves
    rotated_stem = pygame.transform.rotate(stem_img, sway_angle + shake_intensity * random.uniform(-1, 1))
    rotated_leaves = pygame.transform.rotate(leaves_img, sway_angle + shake_intensity * random.uniform(-1, 1))

    # Draw stem
    screen.blit(rotated_stem, (390, 300))

    # Draw leaves
    screen.blit(rotated_leaves, (360, 280))

    # Add flowers or effects based on emotion
    if emotion == "happy":
        # Bloom flowers
        if len(blooming_flowers) < 5:
            blooming_flowers.append((random.randint(350, 450), random.randint(200, 300)))
        for flower in blooming_flowers:
            screen.blit(flower_img, flower)
    elif emotion == "sad":
        # Falling leaves
        if len(falling_leaves) < 10:
            falling_leaves.append((random.randint(350, 450), random.randint(200, 300), random.uniform(-1, 1)))
        for i, leaf in enumerate(falling_leaves):
            x, y, speed = leaf
            y += speed * 2
            if y > screen_height:
                falling_leaves.pop(i)
            else:
                screen.blit(leaves_img, (x, y))

        # Falling petals
        if len(falling_petals) < 15:
            falling_petals.append((random.randint(350, 450), random.randint(200, 300), random.uniform(-1, 1)))
        for i, petal in enumerate(falling_petals):
            x, y, speed = petal
            y += speed * 2
            x += random.uniform(-1, 1)  # Add slight horizontal movement
            if y > screen_height:
                falling_petals.pop(i)
            else:
                screen.blit(petal_img, (x, y))
    elif emotion == "angry":
        # Shake the plant
        shake_intensity = 5
        # Add a red glow
        glow = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 0, 0, 128), (400, 300), 100)
        screen.blit(glow, (0, 0))
    else:
        shake_intensity = 0

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/emotion")
async def get_emotion():
    emotion = detect_emotion()
    # Convert pygame surface to base64 image
    pygame.image.save(screen, "temp_screen.png")
    with open("temp_screen.png", "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    return {
        "emotion": emotion,
        "plant_image": encoded_image
    }

# Main loop
def main():
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Detect emotion
        emotion = detect_emotion()

        # Update the screen
        screen.fill((255, 255, 255))  # Fill screen with white
        draw_plant(emotion)
        pygame.display.flip()

        # Add a small delay to reduce CPU usage
        time.sleep(0.1)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)