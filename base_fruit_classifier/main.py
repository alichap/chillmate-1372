
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
import pathlib

#from fruit_classifier_basic_model import train_basic_model
from .fruit_classifier_basic_model import train_basic_model
#from registry import save_model
from .registry import save_model

def train_save_basic_model():

    #dataset_path = "/Users/andreslemus/code/alichap/chillmate-1372/raw_data/fruit/"
    dataset_path = "./raw_data/fruit/"
    #dataset_path = "gs://chillmate_bucket4"

    model, history = train_basic_model(dataset_path, 32, 180, 180)

    print("HERE ACCURACY AND LOSS")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    print("training accuracy:", acc)
    print("validation accuracy", val_acc)

    print("training loss:", loss)
    print("validation loss", val_loss)

    save_model(model)

    return None


if __name__ == '__main__':
    train_save_basic_model()
