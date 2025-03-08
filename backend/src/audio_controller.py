import pygame
import os

class AudioController:
    def __init__(self):
        pygame.mixer.init(frequency=44100, channels=2)
        self.current_emotion = None
        self.is_playing = False
        self.music_paths = {
            'Happy': '../assets/audio/happy_background.mp3',
            'Sad': '../assets/audio/sad_background.mp3',
            'Angry': '../assets/audio/angry_background.mp3',
            'Neutral': '../assets/audio/neutral_background.mp3'
        }
        
    def play_emotion_music(self, emotion):
        try:
            if emotion != self.current_emotion:
                self.current_emotion = emotion
                music_path = os.path.join(
                    os.path.dirname(__file__), 
                    self.music_paths.get(emotion, self.music_paths['Neutral'])
                )
                
                if os.path.exists(music_path):
                    # If music is already playing, fade it out
                    if pygame.mixer.music.get_busy():
                        pygame.mixer.music.fadeout(1000)
                        pygame.time.wait(1000)  # Wait for fadeout to complete
                    
                    try:
                        pygame.mixer.music.load(music_path)
                        pygame.mixer.music.play(-1)  # Loop indefinitely
                        pygame.mixer.music.set_volume(0.3)  # Set volume to 30%
                        self.is_playing = True
                        print(f"Playing {emotion} music")
                    except pygame.error as e:
                        print(f"Error playing music: {e}")
                else:
                    print(f"Music file not found: {music_path}")
        except Exception as e:
            print(f"Error in play_emotion_music: {e}")
    
    def stop_music(self):
        try:
            if self.is_playing:
                pygame.mixer.music.fadeout(1000)
                self.is_playing = False
                self.current_emotion = None
                print("Music stopped")
        except Exception as e:
            print(f"Error stopping music: {e}")

    def __del__(self):
        try:
            pygame.mixer.quit()
        except:
            pass