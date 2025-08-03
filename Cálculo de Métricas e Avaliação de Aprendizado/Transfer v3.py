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
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Se quiser usar GPU, comente a linha abaixo
# tf.config.set_visible_devices([], 'GPU')

# Descompacta o dataset
zip_path     = r'C:\Users\flawl\Downloads\kagglecatsanddogs_5340.zip'
extract_path = r'C:\Users\flawl\Downloads\kagglecatsanddogs_5340'
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

def split_into_dirs(src_folder, dst_root, test_size=0.1, val_size=0.2):
    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    train_val, test = train_test_split(files, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)

    cls = os.path.basename(src_folder)  # 'Cat' ou 'Dog'
    for subset, names in [('train', train), ('val', val), ('test', test)]:
        dst_dir = os.path.join(dst_root, subset, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in names:
            shutil.copy(os.path.join(src_folder, fname), os.path.join(dst_dir, fname))

# Defina raiz do novo dataset
dst_root = os.path.join(extract_path, 'dataset')

# Gera splits para gatos e cachorros
split_into_dirs(cats_dir, dst_root, test_size=0.2, val_size=0.2)
split_into_dirs(dogs_dir, dst_root, test_size=0.2, val_size=0.2)

# Parâmetros
img_size    = (299, 299)
batch_size  = 256
num_classes = 2
epochs_head = 1
epochs_finetune = 1

# Data augmentation só no treino
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    os.path.join(dst_root, 'train'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_gen = test_val_datagen.flow_from_directory(
    os.path.join(dst_root, 'val'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_gen = test_val_datagen.flow_from_directory(
    os.path.join(dst_root, 'test'),
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
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

# Após o fine-tuning, avalie o modelo no test_gen
print("\n===== Avaliação no conjunto de teste =====")
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# Gera previsões
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

# Exibe relatório
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

# Exibe matriz de confusão
cm = confusion_matrix(y_true, y_pred)
print("\nMatriz de confusão:")
print(cm)

# Métricas de desempenho
precisao = np.mean(y_pred == y_true)
sensibilidade = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
especificidade = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
acuracia = (cm[0, 0] + cm[1, 1]) / np.sum(cm) if np.sum(cm) > 0 else 0
Fscore = 2 * (precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) > 0 else 0
print(f"\nPrecisão: {acuracia:.4f}")
print(f"Sensibilidade: {sensibilidade:.4f}")
print(f"Especificidade: {especificidade:.4f}")
print(f"Acurácia: {acuracia:.4f}")
print(f"F-score: {Fscore:.4f}")

