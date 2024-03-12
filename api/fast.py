import pandas as pd
# $WIPE_BEGIN

from fastapi import FastAPI
from tensorflow.keras.applications.vgg16 import VGG16

app = FastAPI()

def load_model():
    model = VGG16(weights="imagenet", include_top=False, input_shape=(img_height, img_width, 3))
    return model

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
app.state.model = load_model()

# $WIPE_END

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
