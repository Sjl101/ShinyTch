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
data_dir = pathlib.Path("E:\pokemon")

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

bulbasaur = list(data_dir.glob('001-bulbasaur/*'))
ivysaur = list(data_dir.glob('002-ivysaur/*'))
venusaur = list(data_dir.glob('003-venusaur/*'))
charmander = list(data_dir.glob('004-charmander/*'))
charmeleon = list(data_dir.glob('005-charmeleon/*'))
charizard = list(data_dir.glob('006-charizard/*'))
squirtle = list(data_dir.glob('007-squirtle/*'))
wartortle = list(data_dir.glob('008-wartortle/*'))
blastoise = list(data_dir.glob('009-blastoise/*'))
caterpie = list(data_dir.glob('010-caterpie/*'))
metapod = list(data_dir.glob('011-metapod/*'))
butterfree = list(data_dir.glob('012-butterfree/*'))
weedle = list(data_dir.glob('013-weedle/*'))
kakuna = list(data_dir.glob('014-kakuna/*'))
beedrill = list(data_dir.glob('015-beedrill/*'))
pidgey = list(data_dir.glob('016-pidgey/*'))
pidgeotto = list(data_dir.glob('017-pidgeotto/*'))
pidgeot = list(data_dir.glob('018-pidgeot/*'))
rattata = list(data_dir.glob('019-rattata/*'))
raticate = list(data_dir.glob('020-raticate/*'))
spearow = list(data_dir.glob('021-spearow/*'))
fearow = list(data_dir.glob('022-fearow/*'))
ekans = list(data_dir.glob('023-ekans/*'))
arbok = list(data_dir.glob('024-arbok/*'))
pikachu = list(data_dir.glob('025-pikachu/*'))
raichu = list(data_dir.glob('026-raichu/*'))
sandshrew = list(data_dir.glob('027-sandshrew/*'))
sandslash = list(data_dir.glob('028-sandslash/*'))
nidoranf = list(data_dir.glob('029-nidoran-f/*'))
nidorina = list(data_dir.glob('030-nidorina/*'))
nidoqueen = list(data_dir.glob('031-nidoqueen/*'))
nidoranm = list(data_dir.glob('032-nidoran-m/*'))
nidorino = list(data_dir.glob('033-nidorino/*'))
nidoking = list(data_dir.glob('034-nidoking/*'))
clefairy = list(data_dir.glob('035-clefairy/*'))
clefable = list(data_dir.glob('036-clefable/*'))
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
