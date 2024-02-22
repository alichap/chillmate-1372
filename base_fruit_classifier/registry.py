import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

#from params import MODEL_TARGET
#from params import BUCKET_NAME

from .params import MODEL_TARGET
from .params import BUCKET_NAME

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
    LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "chillmate_models")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    #model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "chillmate",f"{timestamp}.h5")
    model_path = os.path.join(LOCAL_REGISTRY_PATH,f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client(project="chillmate_test1")
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None


if __name__ == '__main__':
    pass
