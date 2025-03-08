import pygame
import os

class AudioController:
    def __init__(self):
        pygame.mixer.init()
        self.current_emotion = None
        self.music_paths = {
            'Happy': '../assets/audio/happy_background.mp3',
            'Sad': '../assets/audio/sad_background.mp3',
            'Angry': '../assets/audio/angry_background.mp3',
            'Neutral': '../assets/audio/neutral_background.mp3'
        }
        
    def play_emotion_music(self, emotion):
        if emotion != self.current_emotion:
            self.current_emotion = emotion
            music_path = os.path.join(
                os.path.dirname(__file__), 
                self.music_paths.get(emotion, self.music_paths['Neutral'])
            )
            
            if os.path.exists(music_path):
                pygame.mixer.music.fadeout(1000)  # Fade out current music
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.play(-1)  # Loop indefinitely
                pygame.mixer.music.set_volume(0.3)  # Set volume to 30%
    
    def stop_music(self):
        pygame.mixer.music.fadeout(1000)
        self.current_emotion = None