import os
import zipfile
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Se quiser usar GPU, comente a linha abaixo
# tf.config.set_visible_devices([], 'GPU')

# Descompacta o dataset
zip_path     = r'F:\Downloads\kagglecatsanddogs_5340.zip'
extract_path = r'F:\Downloads\kagglecatsanddogs_5340.'
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as z: 
        z.extractall(extract_path)

base_dir = os.path.join(extract_path, 'PetImages')
cats_dir = os.path.join(base_dir, 'Cat')
dogs_dir = os.path.join(base_dir, 'Dog')

# Remove imagens corrompidas
def remove_corrompidos(folder):
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with Image.open(fpath) as img:
                img.verify()
        except:
            os.remove(fpath)

remove_corrompidos(cats_dir)
remove_corrompidos(dogs_dir)

# Parâmetros
img_size    = (299, 299)
batch_size  = 128
num_classes = 2
epochs_head = 10
epochs_finetune = 10

# Data augmentation
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    base_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Carrega InceptionV3 sem topo e congela todas as suas camadas
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=img_size + (3,))
for layer in inception.layers:
    layer.trainable = False

# Monta nova cabeça
x = GlobalAveragePooling2D()(inception.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inception.input, outputs=output)

# Compile cabeça
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
]

# Treina só a cabeça
history_head = model.fit(
    train_gen,
    epochs=epochs_head,
    validation_data=val_gen,
    callbacks=callbacks
)

# Fine-tuning: libera últimas 50 camadas do Inception
for layer in inception.layers[-50:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# Recompile com LR menor
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treina novamente — ajustando camadas finais
history_finetune = model.fit(
    train_gen,
    epochs=epochs_finetune,
    validation_data=val_gen,
    callbacks=callbacks
)