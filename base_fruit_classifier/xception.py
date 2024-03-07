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
#import cv2
import statistics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
#from google.colab import drive
from keras.models import model_from_json
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16


def train_xception(): # add parameters if needed

    print("Test")
    pass    # add function code


    return # return model



if __name__ == '__main__':
    dataset_path = 'gs://chillmate_tiny_dataset/'
    dataset_bucket_name = "chillmate_tiny_dataset"

    #num_classes = len(get_dataset_classes(dataset_bucket_name))
    train_xception()
