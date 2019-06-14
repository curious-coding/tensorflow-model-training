import os
import signal

import tensorflow as tf
from tensorflow.python.keras.optimizers import RMSprop

from src.Utils import unzip, setup, generator
from src.data.ImageData import Images

data = Images()

# Unpack training and validation images
#unzip(data.training)
#unzip(data.validation)

# Setup training
train_animal_names = setup(data.animals)
train_plant_names = setup(data.plants)

# Setup validation
validation_animal_names = setup(data.valid_animals)
validation_plant_names = setup(data.valid_plants)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu, input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.summary()

model.compile(
    optimizer=RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

train_generator = generator(data.training + "/")
validation_generator = generator(data.validation + "/")

history = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=8
)

os.kill(os.getpid(), signal.SIGKILL)
