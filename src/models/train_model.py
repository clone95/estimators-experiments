import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.keras import datasets, layers, models


data_dir = 'data/processed'
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

CLASS_NAMES = os.listdir(data_dir + '/train/')

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
STEPS_PER_EPOCH = np.ceil(BATCH_SIZE)

print(CLASS_NAMES)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir + '/train'),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH))


val_data_gen = image_generator.flow_from_directory(directory=str(data_dir + '/val'),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs=2)
