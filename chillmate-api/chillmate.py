from fastapi import FastAPI, UploadFile, File
import requests
import uuid
from base_fruit_classifier.main import predict_in_prod_img

app = FastAPI()

# define root
# first endpoint
@app.get("/")
def status():
    return {"API": "connected"}

@app.post("/predict")
async def create_upload_file(file: UploadFile= File(...)):
    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()
    with open(f"data/{file.filename}", "wb") as f:
        f.write(contents)
    img_path = f'data/{file.filename}'
    prediction = predict_in_prod_img(img_path)
    return {"label": prediction
            }
