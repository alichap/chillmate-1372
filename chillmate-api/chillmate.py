from fastapi import FastAPI
import requests
#from base_fruit_classifier.xception-model import train_xception
#from base_fruit_classifier.model_test import forecast
from base_fruit_classifier.main import predict_in_prod_img

app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected"}

  
#@app.get("/predict")
#def predict(X):

    #model = pickle.load_model()
    #prediction = predict_in_prod_img(X)

    #return {'image_predict': prediction}


url = 'http://localhost:8000/predict'

params = {
    'image_predict': 0
}

response = requests.get(url, params=params)
response.json() #=> {wait: 64}


