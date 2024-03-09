
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

from base_fruit_classifier.params import *
from google.cloud import storage
from base_fruit_classifier.fruit_classifier_basic_model import *
from base_fruit_classifier.registry import *

from base_fruit_classifier.resnet50_gcp4 import *
from base_fruit_classifier.vgg16a import *
from base_fruit_classifier.vgg16al import *
from base_fruit_classifier.xception import *



def train_save_basic_model():

    #dataset_path = "/Users/andreslemus/code/alichap/chillmate-1372/raw_data/fruit/"
    #dataset_path = "./raw_data/fruit/"
    #dataset_path = "gs://chillmate_dataset/"
    dataset_path = "gs://chillmate_tiny_dataset/"

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


def train_save_resnet50(epochs=100):
    #dataset_path = "gs://chillmate_dataset/"
    #dataset_bucket_name = "chillmate_dataset"
    #dataset_path = "gs://chillmate_dataset/"
    #dataset_bucket_name = "chillmate_dataset"
    dataset_path = "gs://chillmate-dataset-mix-0306/"
    dataset_bucket_name = "chillmate-dataset-mix-0306"

    num_classes = len(get_dataset_classes(dataset_bucket_name))

    #dataset_path = //BUCKET_DATASET

    model = train_resnet50(dataset_path, num_classes, epochs)

     # Save the model
    save_model(model, "resnet50")

    return None


def train_save_vgg16():
    dataset_path = "gs://chillmate-dataset-mix-0306/"
    dataset_bucket_name = "chillmate-dataset-mix-0306"
    #dataset_path = "gs://chillmate_dataset/"
    #dataset_bucket_name = "chillmate_dataset"
    #dataset_path = "gs://chillmate_tiny_dataset/"
    #dataset_bucket_name = "chillmate_tiny_dataset"
    num_classes = len(get_dataset_classes(dataset_bucket_name))

    #dataset_path = //BUCKET_DATASET

    model = train_vgg16a(dataset_path)

    # Save the model
    save_model(model, "vgg16")

    #model.summary()

    return None


def train_save_vgg16al():
    epochs = 30
    #dataset_path = "gs://chillmate-dataset-mix-0306/train"
    #dataset_bucket_name = "chillmate-dataset-mix-0306/train"
    #dataset_path = "gs://chillmate_dataset/"
    #dataset_bucket_name = "chillmate_dataset"
    #dataset_path = "gs://chillmate_tiny_dataset/"
    #dataset_bucket_name = "chillmate_tiny_dataset"
    dataset_path = "/home/andreslemus/dataset_vegfru"
    #dataset_bucket_name = "/home/andreslemus/dataset_vegfru"

    #num_classes = len(get_dataset_classes(dataset_bucket_name))
    num_classes=292

    #dataset_path = //BUCKET_DATASET

    model = train_vgg16al(dataset_path, num_classes, epochs)

    # Save the model
    save_model(model, "vgg16")

    #model.summary()

    return None




def train_save_xception():
    dataset_path = "gs://chillmate_tiny_dataset/"
    dataset_bucket_name = "chillmate_tiny_dataset"

    num_classes = len(get_dataset_classes(dataset_bucket_name)) # just in case this parameter is needed

    model = train_xception()

     # Save the model
    save_model(model, "xception")

    #model.summary()

    return None


def predict(model_type, img_height, img_width):
    """
    Make a prediction using the latest trained model
    """

    class_names = get_dataset_classes(BUCKET_DATASET)

    batch_size = 32
    img_height = img_height
    img_width = img_width

    model = load_trained_model(model_type)
    download_images_to_predict() # dowload images to predict to local path

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

            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])

            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
                )
        else:
            pass
            #print("Item is not .jpg, .jpeg or .png")
            #return None
    return None


def predict_in_prod():
    '''
        Return a list of strings that correspond to the classes predicted.
        It uses the most recent model specified in the variable model_type
    '''

    class_names = get_dataset_classes(BUCKET_DATASET) # dataset reference classes

    model_type= "resnet50" # model to use
    img_height = 100
    img_width = 100

    model = load_trained_model(model_type)
    download_images_to_predict()
    images_path = os.path.join(LOCAL_REGISTRY_PATH,'images-to-predict')

    predictions_output = []
    # Predict the class for each image in the folder images_to_predict
    for filename in os.listdir(images_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path_image_to_predict = os.path.join(images_path, filename)

            img = tf.keras.utils.load_img(path_image_to_predict, target_size=(img_height, img_width))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = model.predict(img_array, verbose=0)
            #print(predictions)
            score = tf.nn.softmax(predictions[0])

            current_prediction = class_names[np.argmax(score)]
            predictions_output.append(current_prediction)

    return predictions_output


def predict_in_prod_img(img_path):
    '''
        Return a list of strings that correspond to the classes predicted.
        It uses the most recent model specified in the variable model_type
    '''

    #class_names = ['Apple','Aubergine','Avocado','Banana','Bean','Broccoli','Cabbage','Carrots','Corn','Cucumber',
    #                'Egg','Garlic','Leek_lot','Onion','Pear','Pepper','Potato','Salmon','Tomato','Zucchini'] # dataset reference classes

    class_names = get_dataset_classes(BUCKET_DATASET) # dataset reference classes

    model_type= "resnet50" # model to use
    img_height = 100
    img_width = 100

    model = load_trained_model(model_type)

    # Predict image from query user
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    score = tf.nn.softmax(predictions[0])
    current_prediction = class_names[np.argmax(score)]

    return current_prediction


if __name__ == '__main__':
    #train_save_basic_model()
    #loaded_model = load_model() # load most-recent model trained
    #print(loaded_model)
    #car.summary()
    #print(len(get_dataset_classes()))

    #train_save_vgg16()
    #pred_from_gcs()
    #pred_from_gcs()
    #predict(model_type="vgg16", img_height=348, img_width=348)
    #predict(model_type="resnet50", img_height=100, img_width=100)
    #train_save_resnet50(5)
    #train_save_xception()
    #predict(model_type="vgg16", img_height=348, img_width=348)
    print(predict_in_prod())
