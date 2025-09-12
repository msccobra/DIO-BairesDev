import os
import pandas as pd
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
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suprime avisos do TensorFlow

zip_path     = r"F:\Downloads\archive.zip"  # Caminho para o zip do dataset
extract_path = r"F:\Downloads\Extract"      # Pasta onde extrair
base_dir     = os.path.join(extract_path, 'fashion_small')
images_dir = os.path.join(base_dir, 'resized_images')
csv_path   = os.path.join(base_dir, 'styles.csv')
model_path = os.path.join(base_dir, "fashion_classifier_model.h5")

# Descompacta o dataset, se necessário
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as z: 
        z.extractall(extract_path)


df = pd.read_csv(csv_path, on_bad_lines='skip')
df = df[['id', 'subCategory']].dropna()
df['id'] = df['id'].astype(str)
df['filename'] = df['id'] + '.jpg'

contagens = df['subCategory'].value_counts()
amostra_cats = contagens[contagens > 1000].sample(n=4, random_state=random.randint(0, 10_000))
classes = list(amostra_cats.index)
num_classes = len(classes)

print(f"Categorias escolhidas para treinamento (CSV): {classes}")
print(f"Número de classes selecionadas: {num_classes}")

# Mantém apenas linhas com classes escolhidas e arquivos que realmente existem
df = df[df['subCategory'].isin(classes)].copy()
df['exists'] = df['filename'].apply(lambda f: os.path.isfile(os.path.join(images_dir, f)))
df = df[df['exists']].drop(columns='exists')

# Verificação mínima
if df.empty:
    raise RuntimeError("Nenhuma imagem encontrada para as classes selecionadas.")


# Parâmetros de treinamento
img_size    = (299, 299)  # InceptionV3 requer 299x299
batch_size  = 48          # Reduzido para datasets menores
epochs_head = 1          # Aumentado para melhor treinamento inicial
epochs_finetune = 1
learning_rate_head = 1e-3
learning_rate_finetune = 1e-5


# Data augmentation adaptada para roupas/objetos
datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.4,  # 0.4 para separar val (0.2) e test (0.2)
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Generator apenas para validação (sem augmentation)
datagen_val = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

seed = 42  # use um seed fixo se quiser splits reprodutíveis

val_indices, test_indices = train_test_split(
    np.arange(len(df)),
    test_size=0.2,  # 20% para teste
    random_state=seed
)
train_indices, val_indices = train_test_split(
    np.setdiff1d(np.arange(len(df)), test_indices),
    test_size=0.25,  # 25% de 80% = 20% para validação
    random_state=seed
)

df_train = df.iloc[train_indices].copy()
df_val = df.iloc[val_indices].copy()
df_test = df.iloc[test_indices].copy()

train_gen = datagen_train.flow_from_dataframe(
    dataframe=df_train,
    directory=images_dir,
    x_col='filename',
    y_col='subCategory',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    classes=classes, 
    seed=seed
)

val_gen = datagen_val.flow_from_dataframe(
    dataframe=df_val,
    directory=images_dir,
    x_col='filename',
    y_col='subCategory',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    classes=classes,
    seed=seed
)

test_gen = datagen_val.flow_from_dataframe(
    dataframe=df_test,
    directory=images_dir,
    x_col='filename',
    y_col='subCategory',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    classes=classes,
    seed=seed
)


print(f"Amostras de treino: {train_gen.samples}")
print(f"Amostras de validação: {val_gen.samples}")
print(f"Amostras de teste: {test_gen.samples}")

if os.path.isfile(model_path): # Carrega modelo previamente treinado, se existir
    model = load_model(model_path)
else:
    #instancia o InceptionV3 com pesos do ImageNet
    inception = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=img_size + (3,)
    )
    for layer in inception.layers:
        layer.trainable = False
  

# Carrega InceptionV3 pré-treinado sem o topo
inception = InceptionV3(
    weights='imagenet', 
    include_top=False, 
    input_shape=img_size + (3,)
)

# Congela todas as camadas inicialmente
for layer in inception.layers:
    layer.trainable = False

# Constrói nova cabeça classificadora
x = GlobalAveragePooling2D()(inception.output)
x = Dense(512, activation='relu')(x)  # Camada maior para mais capacidade
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation='softmax')(x)


model = Model(inputs=inception.input, outputs=output)

print("\n=== ARQUITETURA DO MODELO ===")
print(f"Entrada: {img_size + (3,)}")
print(f"Backbone: InceptionV3 (congelado)")
print(f"Cabeça: GlobalAvgPool -> Dense(512) -> Dropout(0.5) -> Dense(256) -> Dropout(0.4) -> Dense({num_classes})")
print(f"Total de parâmetros: {model.count_params():,}")

# Compilação para treinamento da cabeça
model.compile(
    optimizer=Adam(learning_rate=learning_rate_head),
    loss='categorical_crossentropy',
    metrics = ['accuracy']
)

# Callbacks
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=4, 
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1
    )
]

print("\n=== FASE 1: TREINAMENTO DA CABEÇA ===")
print("Treinando apenas a nova cabeça classificadora...")

# Treina apenas a cabeça
history_head = model.fit(
    train_gen,
    epochs=epochs_head,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

print(f"\nMelhor acurácia na validação (cabeça): {max(history_head.history['val_accuracy']):.4f}")

print("\n=== FASE 2: FINE-TUNING ===")
print("Liberando últimas camadas do InceptionV3 para fine-tuning...")

# Fine-tuning: libera as últimas camadas do Inception
layers_to_unfreeze = 60  # Ajuste conforme necessário
for layer in inception.layers[-layers_to_unfreeze:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

print(f"Camadas treináveis após fine-tuning: {sum([1 for layer in model.layers if layer.trainable])}")

# Recompila com learning rate menor
model.compile(
    optimizer=Adam(learning_rate=learning_rate_finetune),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treina com fine-tuning
history_finetune = model.fit(
    train_gen,
    epochs=epochs_finetune,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

print(f"\nMelhor acurácia na validação (fine-tuning): {max(history_finetune.history['val_accuracy']):.4f}")

# Salva o modelo treinado
model_save_path = os.path.join(base_dir, 'fashion_classifier_model.keras')
model.save(model_save_path)
print(f"\nModelo salvo em: {model_save_path}")

# Exibe resumo final
print("\n=== RESUMO FINAL ===")
print(f"Dataset: {base_dir}")
print(f"Classes: {num_classes}")
print(f"Amostras treino/validação/teste: {train_gen.samples}/{val_gen.samples}/{test_gen.samples}")
print(f"Acurácia final: {max(history_finetune.history['val_accuracy']):.4f}")

# Função para fazer predições
def predict_image(image_path, model, class_indices):
    """
    Faz predição em uma única imagem
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Inverte o dicionário de classes
    idx_to_class = {v: k for k, v in class_indices.items()}
    predicted_class = idx_to_class[predicted_class_idx]
    
    return predicted_class, confidence

print(f"\nClasses disponíveis: {train_gen.class_indices}")

#Gera previsões
y_pred_probs = model.predict(test_gen, verbose=1) # Fazer no val_gen, pois test_gen não foi definido, por ora
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

# Exibe relatório
print("\nClassification Report:")
y_pred_probs = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes
print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys()), digits=3))


# Função para extrair embeddings
def get_embeddings(model, generator):
    # Cria um modelo que retorna a penúltima camada (antes do softmax)
    embedding_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    embeddings = embedding_model.predict(generator, verbose=1)
    return embeddings

# Extrai embeddings do conjunto de teste
embeddings = get_embeddings(model, test_gen)
filenames = test_gen.filenames
labels = test_gen.classes
class_indices = test_gen.class_indices
idx_to_class = {v: k for k, v in class_indices.items()}

# Seleciona uma imagem de cada classe
selected_indices = []
for class_id in np.unique(labels):
    idx = np.where(labels == class_id)[0][0]  # pega a primeira ocorrência
    selected_indices.append(idx)

def show_recommendations(query_idx, similar_idxs, filenames, labels, idx_to_class, images_dir):
    plt.figure(figsize=(12, 3))
    # Exibe imagem consultada
    plt.subplot(1, 5, 1)
    img_path = os.path.join(images_dir, filenames[query_idx])
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Consulta\n{idx_to_class[labels[query_idx]]}")

    # Exibe recomendações
    for i, sim_idx in enumerate(similar_idxs):
        plt.subplot(1, 5, i+2)
        img_path = os.path.join(images_dir, filenames[sim_idx])
        img = Image.open(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Rec {i+1}\n{idx_to_class[labels[sim_idx]]}")
    plt.tight_layout()
    plt.show()

# Para cada imagem selecionada, recomenda 4 mais similares e as exibe
# Será aberta uma janela com as 5 imagens, a primeira, selecionada aleatoriamente e as 4 similares
# Para que a próxima janela seja aberta, a que está na tela deve ser fechada, essa é uma limitação do matplotlib, a biblioteca plotly seria capaz de fazê-lo simultaneamente.
for idx in selected_indices:
    query_emb = embeddings[idx].reshape(1, -1)
    dists = cosine_distances(query_emb, embeddings)[0]
    similar_idxs = np.argsort(dists)[1:5]
    print(f"\nRecomendações para imagem '{filenames[idx]}' da classe '{idx_to_class[labels[idx]]}':")
    for sim_idx in similar_idxs:
        print(f"  {filenames[sim_idx]} (classe: {idx_to_class[labels[sim_idx]]}, distância: {dists[sim_idx]:.3f})")

    show_recommendations(idx, similar_idxs, filenames, labels, idx_to_class, images_dir)
