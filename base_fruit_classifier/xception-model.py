# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, os.path
import os
#from google.colab import drive
import PIL
from PIL import Image
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Sequential
import statistics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from keras.models import model_from_json
from keras.models import load_model
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint


def train_xception(dataset_path):

    # variable to prepare the dataset
    data_dir_train = dataset_path

    batch_size = 32
    img_height = 348
    img_width = 348

    #create dataset for train
    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    # création d'une variable nom des classes
    class_names = train_ds.class_names
    num_classes = len(class_names)

    #create dataset for evaluation
    eval_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir_train,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)


    #mise en cache des données
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    eval_ds = eval_ds.cache().prefetch(buffer_size=AUTOTUNE)


    #Loading model
    def load_model():
        model = Xception(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3), classifier_activation='softmax')
        return model


    # Set the first layers to be untrainable method
    def set_nontrainable_layers(model):
        model.trainable = False
        return model



    # Add last layers method
    def add_last_layers(model):
        '''Take a pre-trained model, set its parameters as non-trainable, and add additional trainable layers on top'''
        rescale_layer = layers.Rescaling(1./255, input_shape=(img_height, img_width, 3))
        base_model = set_nontrainable_layers(model)
        flatten_layer = layers.Flatten()
        dense_layer_1 = layers.Dense(250, activation='relu')
        dense_layer_2 = layers.Dense(250, activation='relu')
        prediction_layer = layers.Dense(num_classes, activation='softmax')

        model = models.Sequential([
            rescale_layer,
            base_model,
            flatten_layer,
            dense_layer_1,
            dense_layer_2,
            prediction_layer
        ])
        return model



    def build_model():
        model = load_model()
        model = set_nontrainable_layers(model)
        model = add_last_layers(model)

        opt = optimizers.Adam(learning_rate=1e-3)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    optimizer=opt,
                    metrics=['accuracy'])
        return model

    model = build_model()

    return model


## to add as checkpoint:
checkpoint_path = "/content/drive/MyDrive/Le Wagon - 1372/data-set-test-aline/training_1/cp.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 #save_best_only= True,
                                                 #save_weights_only=True,
                                                 #monitor='val_accuracy',
                                                 #mode='max',
                                                 #verbose=1)

#es = EarlyStopping(monitor = 'val_accuracy',
                   #mode = 'max',
                   #patience = 50,
                   #verbose = 1,
                   #restore_best_weights = False)

#epochs=1000


if __name__ == '__main__':
    # Example usage
    dataset_path = 'gs://chillmate_tiny_dataset/'
    dataset_bucket_name = "chillmate_tiny_dataset"

    #num_classes = len(get_dataset_classes(dataset_bucket_name))
