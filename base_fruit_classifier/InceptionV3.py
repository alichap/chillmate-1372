import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import os

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

# Directory containing my images
image_dir = '/Users/andreslemus/code/alichap/chillmate-1372/raw_data/test_set_example'


# Function to predict the class of an image
def predict_image_class(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded_dims) # preprocesses a tensor or numpy array encoding a batch of images
    print("PRINT DIMENSIONS EXPANDED")
    print(img_array.shape)
    print(img_array_expanded_dims.shape)

    # Make a prediction
    prediction = model.predict(img_preprocessed)

    # Decode the prediction
    decoded_prediction = decode_predictions(prediction, top=3)[0]
    print(f"Predictions for {os.path.basename(image_path)}:")
    for i, (imagenet_id, label, score) in enumerate(decoded_prediction):
        print(f"{i+1}: {label} ({score*100:.2f}%)")
    print("\n")


    # Predict the class for each image in the directory
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        predict_image_class(os.path.join(image_dir, filename))
