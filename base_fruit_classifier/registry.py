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




def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    - if MODEL_TARGET='mlflow', also persist it on MLflow instead of GCS (for unit 0703 only) --> unit 03 only
    """

    #LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
    #LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "chillmate_models")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    #model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "chillmate",f"{timestamp}.h5")
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "chillmate-models",f"{timestamp}.h5")
    model.save(model_path)

    print("👍 Model saved on my local machine")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_MODELS)
        #blob = bucket.blob(f"models/{model_filename}")
        blob = bucket.blob(f"{model_filename}") # no longer saving to the subfolder models in the bucket. Not necessary
        blob.upload_from_filename(model_path)

        print("👍 Model saved to GCS")

        return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    #LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "chillmate_models")


    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("👍 Model loaded from local disk")

        return latest_model


    elif MODEL_TARGET == "gcs":

        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client(project=GCP_PROJECT)
        #blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))
        blobs = list(client.get_bucket(BUCKET_MODELS).list_blobs())

        #print("THIS IS THE BLOBS LIST")
        #for i in blobs:
        #    print(i)
        #print("NOW THE LATEST BLOB")
        #latest_blob = max(blobs, key=lambda x: x.updated)
        #print(latest_blob)
        #print("LATEST BLOB PATH TO SAVE")
        #latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
        #print(latest_model_path_to_save)
        #print("AND THIS IS LATEST MODEL")
        #latest_model = keras.models.load_model(latest_model_path_to_save)
        #print(latest_model)

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH,"chillmate-models",latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print(" 👍 Latest model downloaded from cloud storage")

            latest_model_name_fetched = latest_blob.name
            print("The name of the model fetched from GCP is: ", latest_model_name_fetched)

            return latest_model
        except:
            print(f"\n🙁 No model found in GCS bucket {BUCKET_MODELS}")

            return None


    else:
        print("CONNECTION TO GCP NOT YET AVAILABLE. ASK ANDRES")


def load_images_to_predict():
    '''
    Get images to predict from Cloud Storage bicket and store them locally
    '''
    print(Fore.BLUE + f"\nGetting images to predict from GCS..." + Style.RESET_ALL)

    client = storage.Client(project=GCP_PROJECT)
    #blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))
    blobs = list(client.get_bucket(BUCKET_IMAGES_TO_PREDICT).list_blobs())

    try:
        for blob in blobs:
            local_path_to_save_images = os.path.join(LOCAL_REGISTRY_PATH,"images-to-predict", blob.name)
            blob.download_to_filename(local_path_to_save_images)

        print(" 👍 Images to predict successfully downloaded from cloud storage into local")
        return None

    except:
        print(f"\n🙁 No images found in GCS bucket {BUCKET_IMAGES_TO_PREDICT}")

        return None






if __name__ == '__main__':
    pass
