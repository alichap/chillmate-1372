import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import Dropout
from base_fruit_classifier.main import *
from base_fruit_classifier.registry import *

def train_resnet50():
    # Number of classes in dataset Train
    dataset_path = "gs://chillmate_tiny_dataset/"
    dataset_bucket_name = "chillmate_tiny_dataset"
    num_classes = len(get_dataset_classes(dataset_bucket_name))
    #dataset_classes = os.listdir('/Users/andreslemus/code/alichap/chillmate-1372/raw_data/Training')
    #num_classes = len(os.listdir('/Users/andreslemus/code/alichap/chillmate-1372/raw_data/Training'))  #

    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

    # Add new layers on top of the model
    x = base_model.output
    x = GlobalAveragePooling2D()(base_model.output)
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

    # Path to your training data
    #train_dir = '/Users/andreslemus/code/alichap/chillmate-1372/raw_data/Training'

    # Set up the data generator for training
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(dataset_path,
                                                        target_size=(100, 100),  # Adjusted to match your dataset image size
                                                        color_mode='rgb',
                                                        batch_size=32,
                                                        class_mode='categorical',
                                                        shuffle=True)

    # Train the model
    model.fit(train_generator, epochs=5)  # Adjust epochs according to your needs

    save_model(model)

    return None



if __name__ == '__main__':

    train_resnet50()
