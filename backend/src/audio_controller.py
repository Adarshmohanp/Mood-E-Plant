import pygame
import os
from pathlib import Path

class AudioController:
    def __init__(self):
        # Initialize pygame mixer with higher quality
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        self.current_emotion = None
        self.is_playing = False
        
        # Get absolute paths for audio files
        base_path = Path(__file__).parent.parent / 'assets' / 'audio'
        self.music_paths = {
            'Happy': str(base_path / 'happy_background.mp3'),
            'Sad': str(base_path / 'sad_background.mp3'),
            'Angry': str(base_path / 'angry_background.mp3'),
            'Neutral': str(base_path / 'neutral_background.mp3')
        }
        
        # Verify audio files exist
        for emotion, path in self.music_paths.items():
            if not os.path.exists(path):
                print(f"Warning: Audio file for {emotion} not found at {path}")
    
    def play_emotion_music(self, emotion):
        """Play music for the detected emotion"""
        try:
            if emotion != self.current_emotion:
                music_path = self.music_paths.get(emotion)
                
                if not music_path or not os.path.exists(music_path):
                    print(f"Audio file not found for emotion: {emotion}")
                    return
                
                # If music is currently playing, fade it out
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.fadeout(1000)
                    pygame.time.wait(1000)
                
                try:
                    pygame.mixer.music.load(music_path)
                    pygame.mixer.music.play(loops=-1, fade_ms=1000)
                    pygame.mixer.music.set_volume(0.4)  # Set volume to 40%
                    self.current_emotion = emotion
                    self.is_playing = True
                    print(f"Now playing: {emotion} music")
                except pygame.error as e:
                    print(f"Error playing {emotion} music: {e}")
        
        except Exception as e:
            print(f"Error in play_emotion_music: {e}")
    
    def stop_music(self):
        """Stop the currently playing music"""
        try:
            if self.is_playing:
                pygame.mixer.music.fadeout(1000)
                self.is_playing = False
                self.current_emotion = None
                print("Music stopped")
        except Exception as e:
            print(f"Error stopping music: {e}")
    
    def __del__(self):
        """Cleanup when the controller is destroyed"""
        try:
            pygame.mixer.quit()
        except:
            pass