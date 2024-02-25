import os
import numpy as np

#### VARIABLES ####
GCP_PROJECT = os.environ.get("GCP_PROJECT")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_MODELS = os.environ.get("BUCKET_MODELS")
BUCKET_IMAGES_TO_PREDICT = os.environ.get("BUCKET_IMAGES_TO_PREDICT")



#### CONSTANTS ####
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "chillmate")
