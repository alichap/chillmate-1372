
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
from base_fruit_classifier.params import *

#from fruit_classifier_basic_model import train_basic_model
from .fruit_classifier_basic_model import train_basic_model
#from registry import save_model
from .registry import save_model, load_model
from base_fruit_classifier.params import *
from google.cloud import storage



def train_save_basic_model():

    #dataset_path = "/Users/andreslemus/code/alichap/chillmate-1372/raw_data/fruit/"
    #dataset_path = "./raw_data/fruit/"
    dataset_path = "gs://chillmate_dataset/"
    #dataset_path = //BUCKET_DATASET

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


def get_dataset_classes(batch_size=32, img_height=180, img_width=180):
    """
    Get the classes of the dataset the the model was trained on. These are necessary
    to determine the range of classes the prediction belongs to.
    """
    #dataset_path = f"/Users/andreslemus/code/alichap/chillmate-1372/raw_data/fruit/"
    #dataset_path = os.path.join(LOCAL_REGISTRY_PATH,'dataset')
    dataset_path = f"gs://{BUCKET_DATASET}/"

    train_ds, val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="both",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print("There are", len(class_names), " classes in the dataset")
    print(class_names)

    return class_names


def pred():
    """
    Make a prediction using the latest trained model
    """

    class_names = get_dataset_classes()

    batch_size = 32
    img_height = 180
    img_width = 180

    model = load_model()

    #image_path = f"/Users/andreslemus/code/alichap/chillmate-1372/raw_data/test_set_example/{image_to_test}.jpg"
    #image_path = f"../chillmate-1372/raw_data/test_set_example/{image_to_test}.jpg"
    image_path = os.path.join(LOCAL_REGISTRY_PATH,'images-to-predict')
    #image_path = f"gs://{BUCKET_DATASET}/"

    # Predict the class for each image in the directory
    for filename in os.listdir(image_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path_image_to_predict = os.path.join(image_path, filename)

            print(f"\nPredicting {filename}")

            img = tf.keras.utils.load_img(path_image_to_predict, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
                )
        else:
            print("Item is not .jpg, .jpeg or .png")
            #pass
            return None

    return None


def pred_from_gcs():
    """
    Make a prediction using the latest trained model
    """

    class_names = get_dataset_classes()

    batch_size = 32
    img_height = 180
    img_width = 180

    model = load_model()

    #image_path = f"/Users/andreslemus/code/alichap/chillmate-1372/raw_data/test_set_example/{image_to_test}.jpg"
    #image_path = f"../chillmate-1372/raw_data/test_set_example/{image_to_test}.jpg"
    #image_path = os.path.join(LOCAL_REGISTRY_PATH,'images-to-predict')
    image_path = f"gs://{BUCKET_DATASET}/"

    client = storage.Client(project=GCP_PROJECT)
    blobs = list(client.get_bucket(BUCKET_DATASET).list_blobs())

    # Predict the class for each image in the directory
    for blob in blobs:
        if blob.lower().endswith(('.png', '.jpg', '.jpeg')):
            path_image_to_predict = os.path.join(image_path, filename)

            print(f"\nPredicting {filename}")

            img = tf.keras.utils.load_img(path_image_to_predict, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
                )
        else:
            #print("Image is not .jpg, .jpeg or .png")
            pass
            #return None

    return None




if __name__ == '__main__':
    #train_save_basic_model()
    loaded_model = load_model() # load most-recent model trained
    print(loaded_model)
    #car.summary()



'''import os

# Absolute path
abs_path = f"/Users/andreslemus/code/alichap/chillmate-1372/raw_data/test_set_example/{image_to_test}.jpg"

# Current directory or any directory from which you want to calculate the relative path
current_dir = "/Users/andreslemus/code/alichap/project"

# Convert to relative path
relative_path = os.path.relpath(abs_path, current_dir)

print(relative_path)

f"../chillmate-1372/raw_data/test_set_example/{image_to_test}.jpg"'''