from fastapi import FastAPI, UploadFile, File, Query
import requests
import uuid
from base_fruit_classifier.main import *
from recipe_data_base import *
from recipe_proposal import *
from typing import List

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

    return {#"label": prediction,
            "img": prediction
            }

# @app.get("/recipe")
# async def get_recipe(recipe_dict:dict):
#     #dict_recipe = dict_data_base_setup()
#     #list_ing= list(dict_ing.values())
#     #recipe_list=find_common_recipe(list_ing, dict_recipe)
#     #return {"recipe": "ok"
#             #}
#     return {"recipe": recipe_dict}

@app.get("/recipe")
async def get_recipe(ingredients: List[str] = Query(...)):
    dict_recipe = dict_data_base_setup()
    # Process the ingredients list to generate the recipe
    recipe = find_common_recipe(ingredients, dict_recipe)
    return {"recipe": recipe}

def generate_recipe(ingredients: List[str]) -> str:
    # Example function to generate recipe based on ingredients
    return f"Recipe with {', '.join(ingredients)}"
