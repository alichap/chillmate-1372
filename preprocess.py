# Les imports
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from keras.preprocessing.image import array_to_img


# méthode pour croper image
def crop_center(image_raw, crop_width=348, crop_height=348):

    '''L'image est réduite à une dimensions 800 sur 600.'''
    image_resized = image_raw.resize((800, 600))
    img_width, img_height = image_resized

    '''L'image est cropée au centre.'''
    return image_resized.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))
