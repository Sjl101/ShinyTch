import matplotlib.pyplot as plt
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import PIL
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import pathlib
dataset_url = "https://github.com/Sjl101/Sjl101.github.io/raw/master/Downloads/pokemon.tgz"
data_dir = tf.keras.utils.get_file('pokemon', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

bulbasaur = list(data_dir.glob('001-bulbasaur/*'))
ivysaur = list(data_dir.glob('002-ivysaur/*'))
venusaur = list(data_dir.glob('003-venusaur/*'))
charmander = list(data_dir.glob('004-charmander/*'))
charmeleon = list(data_dir.glob('005-charmeleon/*'))
charizard = list(data_dir.glob('006-charizard/*'))
psyduck = list(data_dir.glob('054-psyduck/*'))


batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

tf.data.experimental.save(
    train_ds, "", compression=None, shard_func=None, checkpoint_args=None
)

num_classes = 898

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

sunflower_url = "https://1.bp.blogspot.com/-YAf8e-hBXxc/UTffQLfg26I/AAAAAAAAFtg/60i0Sra22sE/s1600/Charmeleon.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)

img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

tf.saved_model.save(model, "datasets/")
