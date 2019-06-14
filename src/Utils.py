import os
import zipfile

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


def unzip(path: str):
    local_zip = path + ".zip"
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall(path)
    zip_ref.close()


def setup(path: str):
    training_dir = os.path.join(path)
    return os.listdir(training_dir)


def generator(path: str, mode: str = 'binary'):
    return ImageDataGenerator(rescale=1 / 255).flow_from_directory(
        path,
        target_size=(300, 300),
        batch_size=128,
        class_mode=mode
    )
