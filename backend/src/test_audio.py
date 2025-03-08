from audio_controller import AudioController
import time

def test_audio():
    try:
        audio = AudioController()
        
        # Test each emotion
        emotions = ['Happy', 'Sad', 'Angry', 'Neutral']
        
        for emotion in emotions:
            print(f"\nTesting {emotion} music...")
            audio.play_emotion_music(emotion)
            time.sleep(3)  # Play for 3 seconds
        
        print("\nAudio test completed successfully")
        audio.stop_music()
        
    except Exception as e:
        print(f"Error during audio test: {e}")

if __name__ == "__main__":
    test_audio()