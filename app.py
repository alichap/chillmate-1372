# Les imports
import streamlit as st
import numpy as np
import PIL
from PIL import Image

'''
# CHILLMATE your next recipe buddy.
'''
img_width=800
img_height=600
crop_width=348
crop_height=348

list = []
validation_ingredients = False
go = False

'''
## Les aliments contenus dans votre frigo.
'''

uploaded_files = st.file_uploader("Upload your fridge images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files is not None:

    for uploaded_file in uploaded_files:
        raw_image = Image.open(uploaded_file)
        raw_image = raw_image.save('first_image.jpg')
        st.text('Voici l\'image que vous avez chargée')
        st.image('first_image.jpg')

        raw_image = Image.open('first_image.jpg')
        image_resized = raw_image.resize((800, 600))

        '''Votre image sera rognée de cette façon :'''
        image_resized = image_resized.crop(((img_width - crop_width) // 2,
                            (img_height - crop_height) // 2,
                            (img_width + crop_width) // 2,
                            (img_height + crop_height) // 2)
                        )
        st.image(image_resized)

        list.append('Tomato')
        list.append('Banana')

'''
## Les aliments qui ne sont pas contenus dans votre frigo.
'''

st.write('Veuillez sélectionner la liste des condiments à votre disposition pour compléter les aliments contenus dans votre frigo :')

salt = st.checkbox('Salt')
pepper = st.checkbox('Pepper')
olive_oil = st.checkbox('Olive oil')
sugar = st.checkbox('Sugar')
flour = st.checkbox('Flour')

if salt:
    list.append('Salt')
if pepper:
    list.append('Pepper')
if olive_oil:
    list.append('Olive oil')
if sugar:
    list.append('Sugar')
if flour:
    list.append('Flour')

'''
## Validez les ingrédients sélectionnés.
'''

if list is not None:
    st.write('Voici la liste des ingrédients disponibles :')
    for ingredient in list:
        st.write(ingredient)

if st.button('valider la liste des ingrédients'):
    validation_ingredients = True
    st.write('La liste des ingrédients est validée.')

'''
## Lancer la recherche de recette.
'''

if st.button('Cliquez pour découvrir votre recette'):
    st.write('Voici les 3 recettes que nous vous proposons :')
    st.write('- a')
    st.write('- b')
    st.write('- c')


#params = dict(
#    pickup_datetime=pickup_datetime,
#    pickup_longitude=pickup_longitude,
#    pickup_latitude=pickup_latitude,
#    dropoff_longitude=dropoff_longitude,
#    dropoff_latitude=dropoff_latitude,
#    passenger_count=passenger_count)

#wagon_cab_api_url = 'https://taxifare.lewagon.ai/predict'
#response = requests.get(wagon_cab_api_url, params=params)

#prediction = response.json()

#pred = prediction['fare']

#st.header(f'Fare amount: ${round(pred, 2)}')
