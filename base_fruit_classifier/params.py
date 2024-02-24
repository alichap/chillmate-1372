import os
import numpy as np

#### VARIABLES ####
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")




#### CONSTANTS ####
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "chillmate_models")
