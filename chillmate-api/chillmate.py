from fastapi import FastAPI, UploadFile, File
import requests
import uuid
from base_fruit_classifier.main import *
from recipe_data_base import *
from recipe_proposal import *

app = FastAPI()
#list_cnn = ['spinach','chicken']
# define root
# first endpoint

dict_ing={
"1":"tomatoes",
"2":"chicken",
"3":"carrots"
}
@app.get("/")
def status():
    return {"API": "connected"
            #"recipe_data_base": dict_recipe #dict of recipe ing as key, recipe title as value
           # "recipe list": recipe_list
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
            "img": prediction
            }

@app.get("/recipe")
def get_recipe ():
    dict_recipe = dict_data_base_setup()
    list_ing= list(dict_ing.values())
    recipe_list=find_common_recipe(list_ing, dict_recipe)
    return {"recipe": recipe_list
            }
