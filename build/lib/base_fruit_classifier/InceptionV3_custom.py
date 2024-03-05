import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import os

# Load pre-trained InceptionV3 model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Directory containing my test images
image_dir = '/Users/andreslemus/code/alichap/chillmate-1372/raw_data/test_set_example'

# freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False


# Add new layers for the specific number of classes in the fruit/vegetables dataset
x = base_model.output
x = GlobalAveragePooling2D()(x)  # add a global spatial average pooling layer
x = Dense(1024, activation='relu')(x)  # add a fully-connected layer



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



# custom1
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load the InceptionV3 model without the top layer (fully connected layers)
base_model = InceptionV3(weights='imagenet', include_top=False)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers for your specific number of classes
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
# Add a logistic layer for your classes (replace `your_num_classes` with the number of your classes)
predictions = Dense(your_num_classes, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model (should be done after setting layers to non-trainable)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Load your dataset here
# Make sure to preprocess your inputs and convert your labels to one-hot encoding

# Train the model on the new data for a few epochs
# model.fit(your_dataset, epochs=your_num_epochs, batch_size=your_batch_size)

# Optionally, you might want to fine-tune the model by unfreezing some of the top layers of the base model and retraining with a very low learning rate








# custimsed 2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Load the InceptionV3 model without the top layer
base_model = InceptionV3(weights='imagenet', include_top=False)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add new layers for your specific number of classes
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global spatial average pooling layer
x = Dense(1024, activation='relu')(x)  # Add a fully-connected layer
predictions = Dense(19, activation='softmax')(x)  # Adjusted for 19 fruit classes

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Dataset preparation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2  # Use 20% of the images as a validation set
)

train_generator = train_datagen.flow_from_directory(
    'raw_data/fruit',  # This should be the path to your training data
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    'raw_data/fruit',  # This should be the path to your training data
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Train the model
model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs depending on your dataset size and desired accuracy
    validation_data=validation_generator
)

# Save the model
model.save('base_fruit_classifier/InceptionV3_custom_model.h5')
