import tensorflow as tf
import os

def convert_model():
    try:
        # Path to your existing model (could be .h5 or other format)
        old_model_path = input("Enter path to existing model file: ")
        if not os.path.exists(old_model_path):
            raise FileNotFoundError(f"Model file not found: {old_model_path}")

        # Load the model
        model = tf.keras.models.load_model(old_model_path)

        # Create the output directory if it doesn't exist
        os.makedirs('../assets/models', exist_ok=True)

        # Save in TensorFlow 2.x format
        new_model_path = os.path.join('../assets/models', 'facemodel.keras')
        model.save(new_model_path, save_format='tf')
        print(f"Model converted and saved to: {new_model_path}")

    except Exception as e:
        print(f"Error converting model: {e}")

if __name__ == "__main__":
    convert_model()