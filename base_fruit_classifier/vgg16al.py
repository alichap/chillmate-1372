import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

def train_vgg16al(dataset_path, num_classes, epochs):
    # Define the input shape
    img_height, img_width = 348, 348

    # Load the pre-trained VGG16 model without the top layer
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    # Rescaling layer
    rescale_layer = layers.Rescaling(1./255)

    # Data augmentation layer
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # Add new layers on top of the model
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = data_augmentation(inputs)
    x = rescale_layer(x)
    x = base_model(x, training=False)  # Make sure the base_model runs in inference mode here
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs, outputs)

    # Freeze all layers in the base VGG16 model
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
        image_size=(img_height, img_width),
        batch_size=32,
        label_mode='categorical'
    ).prefetch(buffer_size=AUTOTUNE)

    validation_dataset = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=32,
        label_mode='categorical'
    ).prefetch(buffer_size=AUTOTUNE)

    # Train the model with early stopping
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=[early_stopping, tensorboard_callback]
    )

    return model




if __name__ == '__main__':
    # Example usage
    dataset_path = 'gs://chillmate_tiny_dataset/'
    dataset_bucket_name = "chillmate_tiny_dataset"

    num_classes = len(get_dataset_classes(dataset_bucket_name))
    train(dataset_path, num_classes)
