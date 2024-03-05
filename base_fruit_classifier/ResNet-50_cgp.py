import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Number of classes in your dataset
num_classes = 131  # Change this to the actual number of fruits and vegetable classes you have

# Load the pre-trained ResNet50 model without the top layer
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Add new layers on top of the model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
x = Dense(256, activation='relu')(x)  # Add a fully-connected layer
predictions = Dense(num_classes, activation='softmax')(x)  # Add a logistic layer for 'num_classes' classes

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all convolutional ResNet50 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Path to your training data
train_dir = '/Users/andreslemus/code/alichap/chillmate-1372/raw_data/Training'

# Set up the data generator for training
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(100, 100),  # Adjusted to match your dataset image size
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=True)

# Train the model
model.fit(train_generator, epochs=7)  # Adjust epochs according to your needs

# Function to predict the class of an image using the newly trained model
def predict_image_class(model, image_path):
    img = image.load_img(image_path, target_size=(100, 100))  # Adjusted to match your dataset image size
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array_expanded_dims)
    prediction = model.predict(img_preprocessed)

    # Since we don't have the ImageNet classes, you will need to map the predictions
    # to your classes based on the index
    predicted_class = np.argmax(prediction, axis=1)
    print(f"Predictions for {os.path.basename(image_path)}: Class ID {predicted_class[0]} with confidence {np.max(prediction)*100:.2f}%")
    print("\n")

# Directory containing images to predict
prediction_image_dir = '/Users/andreslemus/code/alichap/chillmate-1372/raw_data/images-to-predict'

# Predict the class for each image in the prediction directory
for filename in os.listdir(prediction_image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        predict_image_class(model, os.path.join(prediction_image_dir, filename))
