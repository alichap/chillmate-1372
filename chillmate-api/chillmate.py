from fastapi import FastAPI
import pickle
#from base_fruit_classifier.xception-model import train_xception
from base_fruit_classifier.model_test import forecast

app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected"}

@app.get("/predict")
def predict(X):

    #model = pickle.load_model()
    prediction = forecast(X)

    return {'forecast': prediction}
