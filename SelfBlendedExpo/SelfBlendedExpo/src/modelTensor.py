import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB4
from keras.preprocessing.image import ImageDataGenerator
import keras


# Define the EfficientNetB4 model for binary classification
def tensorModel(input_shape=(380, 380, 3)):
    base_model = EfficientNetB4(weights='efficientnetb4_notop.h5', include_top=False, input_shape=input_shape)
      # Freeze the base model

    # Create the model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  # Global average pooling to reduce spatial dimensions
        layers.Dense(1024, activation='relu'),  # Fully connected layer
        layers.Dropout(0.5),  # Dropout layer for regularization
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Model summary

