import tensorflow as tf
import os

def create_emotion_model():
    model = tf.keras.Sequential([
        # First Convolution Block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolution Block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Third Convolution Block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Fourth Convolution Block
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Dense Layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax')  # 4 emotions
    ])
    
    return model

if __name__ == "__main__":
    try:
        # Create model
        model = create_emotion_model()
        
        # Create directory if it doesn't exist
        os.makedirs('../assets/models', exist_ok=True)
        
        # Save model
        model_path = os.path.join(os.path.dirname(__file__), '../assets/models/facemodel.keras')
        model.save(model_path, save_format='tf')
        print(f"Model created and saved successfully at: {model_path}")
        
        # Print model summary
        model.summary()
        
    except Exception as e:
        print(f"Error creating model: {e}")