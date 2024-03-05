import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

#from params import MODEL_TARGET
#from params import BUCKET_NAME

#from .params import MODEL_TARGET
#from .params import BUCKET_NAME

from base_fruit_classifier.params import *

#from taxifare.params import *
#import mlflow
#from mlflow.tracking import MlflowClient


def count_items_in_bucket_dataset():
    client = storage.Client(project=GCP_PROJECT)
    blobs = list(client.get_bucket(BUCKET_DATASET).list_blobs())
    print("There are",len(blobs), "items in the bucket dataset")

    return


if __name__ == '__main__':
    #pass

    #dataset_path = "gs://chillmate_tiny_dataset/"
    #dataset_bucket_name = "chillmate_tiny_dataset"
    #get_dataset_classes(dataset_bucket_name)
    #print(get_dataset_classes(dataset_bucket_name))

    #model = load_model_trained(model_type="vgg16")
    #model.summary()

    count_items_in_bucket_dataset()
