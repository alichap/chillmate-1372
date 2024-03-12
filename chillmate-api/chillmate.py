from fastapi import FastAPI, UploadFile, File
import requests
import uuid
from base_fruit_classifier.main import *
from recipe_data_base import *
from recipe_proposal import *

app = FastAPI()
list_cnn = ['spinach','chicken']
# define root
# first endpoint
@app.get("/")
def status():
    dict_recipe = dict_data_base_setup()
    recipe_list=find_common_recipe(list_cnn, dict_recipe)
    return {"API": "connected",
            #"recipe_data_base": dict_recipe #dict of recipe ing as key, recipe title as value
            "recipe list": recipe_list
            }


@app.post("/predict")
async def create_upload_file(file: UploadFile= File(...)):
    file.filename = f'{uuid.uuid4()}.jpg'
    contents = await file.read()
    with open(f"data/{file.filename}", "wb") as f:
        f.write(contents)
    img_path = f'data/{file.filename}'
    prediction = predict_in_prod_img(img_path)

    return {#"label": prediction,
            "img": img_path
            }

#@app.get("/recipe")
#def get_recipe ()
