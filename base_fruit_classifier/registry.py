import glob
import os
import time
import pickle
from PIL import Image
import io

from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage

from base_fruit_classifier.params import *


def save_model(model: keras.Model = None, model_type=None) -> None:
    '''
    Saves model to local or GCP
    '''

    if model_type not in ["resnet50", "vgg16", "basic", "xception"]:
        print("Model type entered not saveable in GCP bucket")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    #model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "chillmate",f"{timestamp}.h5")
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "chillmate-models", model_type,f"{timestamp}_{model_type}.keras")
    model.save(model_path)

    print("👍 Model saved on my local machine")

    if MODEL_TARGET == "gcs":

        model_filename = model_path.split("/")[-1] #
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(BUCKET_MODELS)
        #blob = bucket.blob(f"models/{model_filename}")
        blob = bucket.blob(f"{model_type}/{model_filename}")
        blob.upload_from_filename(model_path)

        print("👍 Model saved to GCS")

        return None


def load_trained_model(model_type) -> keras.Model:
    """

    """

    #LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "chillmate_models")


    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("👍 Model loaded from local disk")

        return latest_model


    elif MODEL_TARGET == "gcs":

        print(Fore.BLUE + f"\nLoading latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client(project=GCP_PROJECT)
        #blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))
        blobs = list(client.get_bucket(BUCKET_MODELS).list_blobs(prefix=model_type))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH,"chillmate-models",latest_blob.name)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)


            print("👍 Latest model downloaded from cloud storage")

            latest_model_name_fetched = latest_blob.name
            print("The model fetched is:", latest_model_name_fetched)

            return latest_model
        except:
            print(f"\n🙁 No model found in GCS bucket {BUCKET_MODELS}")

            return None


    else:
        print("CONNECTION TO GCP NOT YET AVAILABLE. ASK ANDRES")
        return None



def download_training_dataset():
    '''
    Get images to train the model from Cloud Storage bucket dataset and store them locally.
    If some folders or images already exist in destination, it overwrites them all.
    '''
    print(Fore.BLUE + f"\nGetting dataset images from GCS..." + Style.RESET_ALL)

    client = storage.Client(project=GCP_PROJECT)
    #blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))
    blobs = list(client.get_bucket(BUCKET_DATASET).list_blobs())
    blobs_count = len(blobs)

    # Ensure that the base directory exists
    base_dir = os.path.join(LOCAL_REGISTRY_PATH, "dataset")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    try:
        for blob in blobs:
            local_path_to_save_images = os.path.join(LOCAL_REGISTRY_PATH,"dataset", blob.name)
            os.makedirs(os.path.dirname(local_path_to_save_images), exist_ok=True) # Create directories as needed
            blob.download_to_filename(local_path_to_save_images)

        print(f"👍 Successfully downloaded {blobs_count} items from cloud storage into local")
        return None

    except:
        print(f"\n🙁 No images found in GCS bucket {BUCKET_DATASET}")

        return None


def download_images_to_predict():
    '''
    Get images to predict from Cloud Storage bucket and store them locally
    '''
    print(Fore.BLUE + f"\nGetting images to predict from GCS..." + Style.RESET_ALL)

    client = storage.Client(project=GCP_PROJECT)
    #blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))
    blobs = list(client.get_bucket(BUCKET_IMAGES_TO_PREDICT).list_blobs())

    images = []

    try:
        for blob in blobs:
            local_path_to_save_images = os.path.join(LOCAL_REGISTRY_PATH,"images-to-predict", blob.name)
            blob.download_to_filename(local_path_to_save_images)

            # After downloading, open the image and append it to the images list
            with open(local_path_to_save_images, 'rb') as image_file:
                image = Image.open(image_file)
                image.load()  # Make sure PIL has read the image data
                images.append(image)

        print(" 👍 Images to predict successfully downloaded from cloud storage into local")
        return images



    except:
        print(f"\n🙁 No images found in GCS bucket {BUCKET_IMAGES_TO_PREDICT}")

        return None


def count_items_in_bucket_dataset():
    client = storage.Client(project=GCP_PROJECT)
    #blobs = list(client.get_bucket(BUCKET_DATASET).list_blobs())
    blobs = list(client.get_bucket(BUCKET_DATASET).list_blobs())
    print("There are",len(blobs), "items in the bucket dataset")

    return


def print_items_in_bucket_dataset():
    client = storage.Client(project=GCP_PROJECT)
    blobs = list(client.get_bucket(BUCKET_DATASET).list_blobs())
    for i in blobs:
        print(i.name)

    return


def print_items_in_models_dataset():
    client = storage.Client(project=GCP_PROJECT)
    blobs = list(client.get_bucket(BUCKET_MODELS).list_blobs())
    for i in blobs:
        print(i.name)

    return



def get_dataset_classes(dataset_bucket_name):
    """
    Get the classes of the dataset the model was trained on. These are necessary
    to determine the range of classes the prediction belongs to.
    """

    #dataset_path = "gs://chillmate_dataset/"

    storage_client = storage.Client()
    bucket = storage_client.bucket(dataset_bucket_name)
    iterator = bucket.list_blobs(prefix="", delimiter='/')
    prefixes = set()

    for page in iterator.pages:
        prefixes.update(page.prefixes)

    # Extract the class names from the prefixes
    class_names = [prefix.split('/')[-2] for prefix in prefixes if prefix.endswith('/')]
    print("There are:", len(class_names), "classes")
    #print(class_names)

    return class_names





if __name__ == '__main__':
    #pass

    #dataset_path = "gs://chillmate_tiny_dataset/"
    #dataset_bucket_name = "chillmate_tiny_dataset"
    #get_dataset_classes(dataset_bucket_name)
    #print(get_dataset_classes(dataset_bucket_name))

    model = load_trained_model(model_type="resnet50")
    model.summary()

    #count_items_in_bucket_dataset()
    #images1 = download_images_to_predict()
    #for i in images1:
    #    print(i)
