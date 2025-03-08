import tensorflow as tf
import os

def create_emotion_model():
    # Enhanced model architecture for better emotion detection
    model = tf.keras.Sequential([
        # Input Normalization
        tf.keras.layers.InputLayer(input_shape=(48, 48, 1)),
        tf.keras.layers.Rescaling(1./255),
        
        # First Convolution Block - Feature Detection
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Second Convolution Block - Pattern Recognition
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Third Convolution Block - Higher Level Features
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        # Feature Processing
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        
        # Output Layer
        tf.keras.layers.Dense(4, activation='softmax')  # 4 emotions
    ])
    
    # Use a lower learning rate and momentum optimizer
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.9,
        nesterov=True
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
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