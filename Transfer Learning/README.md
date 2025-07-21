# Classificador de Gatos e Cães com InceptionV3 (Transfer v2)

Este projeto treina um modelo baseado em InceptionV3 para fazer a diferenciação entre imagens de diferentes classes, utilizando técnicas de data augmentation e fine-tuning. O dataset utilizado consiste em um banco de dados de aproximadamente 24000 imagens de cães e gatos, e pode ser baixado diretamente do [link](https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip).

---

## Requisitos

Antes de executar, verifique se você possui:

- Python 3.7 ou superior (eu usei 3.10.11)
- Espaço em disco suficiente para o dataset (~1.2 GB)
- O mínimo de 16 GB de memória RAM (com batches de tamanho de 128 imagens, cerca de 13 GB serão usados, batches maiores acelerarão o treionamento, mas usarão mais RAM, e.g., um batch de tamanho 256 usará 16 GB de RAM, contando todos os processos do Windows)
- (Opcional) GPU compatível com CUDA para aceleração (existe uma linha comentada no código que ativará a aceleração por CUDA, como minha GPU é AMD, não foi utilizada essa parte do código) 

Bibliotecas necessárias:

- Numpy
- PIL (Python Image Library)
- Tensorflow >= 2.4
- Zipfile (opcional, eu optei por usar o arquivo .zip diretamente da pasta Downloads)

---

## Como Usar

   O script irá:
   - Descompactar o dataset (se ainda não estiver extraído)  
   - Remover imagens corrompidas  
   - Fazer o processo de Data Augmentation no dataset (basicamente são cortes, rotações e inversões nas imagens originais) 
   - Irá gerar sets de treinamwento, validação e teste (As proporções padrão são 60/20/20, mas isso pode ser facilmente ajustado)
   - Usará a rede de treinamento Inceptionv3
   - Gerará uma saída como a seguinte:

82/82 ━━━━━━━━━━━━━━━━━━━━ 709s 9s/step - accuracy: 0.8644 - loss: 0.3662 - val_accuracy: 0.9908 - val_loss: 0.0265 - learning_rate: 0.0010
Restoring model weights from the end of the best epoch: 1.

82/82 ━━━━━━━━━━━━━━━━━━━━ 680s 8s/step - accuracy: 0.9823 - loss: 0.0495 - val_accuracy: 0.9937 - val_loss: 0.0188 - learning_rate: 1.0000e-05
Restoring model weights from the end of the best epoch: 1.

===== Avaliação no conjunto de teste =====
20/20 ━━━━━━━━━━━━━━━━━━━━ 97s 5s/step - accuracy: 0.9918 - loss: 0.0206 
Test loss:     0.0196
Test accuracy: 0.9930
20/20 ━━━━━━━━━━━━━━━━━━━━ 99s 5s/step 

Classification Report:
              precision    recall  f1-score   support

         Cat       0.99      0.99      0.99      2500
         Dog       0.99      0.99      0.99      2500

    accuracy                           0.99      5000
   macro avg       0.99      0.99      0.99      5000
weighted avg       0.99      0.99      0.99      5000


Confusion Matrix:

[[2479   21]

 [  14 2486]]

 O que significa:

|                | Predição: Negativo | Predição: Positivo |
|----------------|-------------------|-------------------|
| **Real: Negativo** | 2479              | 21                |
| **Real: Positivo** | 14                | 2486              |

---

## Parâmetros Ajustáveis no modelo

| Parâmetro        | Descrição                                         | Valor Padrão   |
|------------------|---------------------------------------------------|----------------|
| `batch_size`     | Tamanho do batch (valores maiores usam maior quantidade de RAM) | 128            |
| `epochs_head`    | Épocas de treinamento da nova “cabeça”            | 10             |
| `epochs_finetune`| Épocas de fine-tuning                             | 10             |
| `val_size`| Proporção do dataset usada para validação no ImageDataGenerator     | 0.2            |
| `test_size`| Proporção do dataset usada para teste no ImageDataGenerator     | 0.2            |


