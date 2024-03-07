from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

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

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

def train_resnet50(dataset_path, num_classes, epochs):
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

    # Add new layers on top of the model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1000, activation='relu')(x)  # Use a single dense layer with 1000 neurons
    predictions = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze all convolutional ResNet50 layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min', restore_best_weights=True)


    # TensorBoard callback
    logdir = "logs/adam/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)


    # Configure the dataset for performance
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # Create a dataset from the images in the dataset_path
    train_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(100, 100),
        batch_size=32,
        label_mode='categorical'
    ).prefetch(buffer_size=AUTOTUNE)

    validation_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(100, 100),
        batch_size=32,
        label_mode='categorical'
    ).prefetch(buffer_size=AUTOTUNE)

    # Train the model with early stopping
    model.fit(
        train_dataset,
        epochs=epochs, # from arguments
        validation_data=validation_dataset,
        callbacks=[early_stopping, tensorboard_callback]
    )

    return model




if __name__ == '__main__':
    # Example usage
    dataset_path = 'gs://chillmate_tiny_dataset/'
    dataset_bucket_name = "chillmate_tiny_dataset"

    num_classes = len(get_dataset_classes(dataset_bucket_name))
    train_resnet50(dataset_path, num_classes)
