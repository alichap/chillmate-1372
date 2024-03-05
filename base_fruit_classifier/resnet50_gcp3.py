import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import Dropout
from base_fruit_classifier.main import *
from base_fruit_classifier.registry import *

def train_resnet50(dataset_path, num_classes):
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

    # Add new layers on top of the model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)  # Increased number of neurons
    x = Dropout(0.5)(x)  # Add dropout
    x = Dense(256, activation='relu')(x)  # Additional dense layer
    predictions = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze all convolutional ResNet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Create a dataset from the images in the GCS bucket
    BATCH_SIZE = 32
    IMG_SIZE = (100, 100)
    train_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    validation_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical'
    )

    # Configure the dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

    # Train the model
    model.fit(train_dataset, epochs=100, validation_data=validation_dataset)  # Adjust epochs according to your needs



    return model



if __name__ == '__main__':
    # Example usage
    dataset_path = 'gs://chillmate_tiny_dataset/'
    dataset_bucket_name = "chillmate_tiny_dataset"

    num_classes = len(get_dataset_classes(dataset_bucket_name))
    train_resnet50(dataset_path, num_classes)
