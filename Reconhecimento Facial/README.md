# Criando um sistema de reconhecimento facial

O desafio da vez foi a criação de um sistema de reconhecimento facial do zero, literalmente: "O objetivo principal deste projeto é trabalhar com as bibliotecas e frameworks estudados e analisados em nossas aulas. Neste sentido, a proposta padrão envolve um sistema de detecção e reconhecimento de faces, utilizando o framework TensorFlow em conjuntos com as bibliotecas que o projetista julgue necessárias, de forma ilimitada.".

Dessa maneira, foram desenvolvidos alguns programas de reconhecimento, os dois melhores estão disponíveis para download (também na pasta relativa a esse projeto), [link1](https://github.com/msccobra/DIO-BairesDev/blob/main/Reconhecimento%20Facial/facial%20v6.py) e [link2](https://github.com/msccobra/DIO-BairesDev/blob/main/Reconhecimento%20Facial/facial%20v8.py). Nessa jornada, muitos problemas foram encontrados no caminho, especialmente a quetsão do estouro de RAM, pois as imagens estavam ficando armazenadas e não estavam sendo liberadas. Em um dataset como o CelebA, que tem cerca de 200k imagens, após cerca de 9,5k, meus 32 GB estouravam. Apesar de tudo, o programa rodava muito rápido, uma vitória inútil. Esses estouros eram causados pela maneira como o pacote MTCNN, que estava sendo usado, funciona. O Tensorflow, por design, mantém um cache do alocador (BFC allocator) e não devolve memória ao SO até o processo terminar, então o uso de RAM cresce enquanto o programa roda. Existem algumas maneiras de mitigar esse problema, mas elas não foram muito eficazes, dado o tamanho do dataset.

Enfim, foi necessária uma mudança completa da abordagem para que esse problema não ocoresse. O programa do primeiro link, infelizmente, não cumpria os requisitos do projeto, pois não foi usado o framework do Tansorflow em qualquer de suas partes, ou mesmo bibliotecas padrão para esse tipo de tarefa, como a cv2 e a ultralytics, abordadas no curso. Dessa maneira foi necessária uma reengenharia completa para que os requisitos fossem cumpridos.

O programa em questão fez uso do framwework Yolov8 para o reconhecimento das faces, baseado nos datasets [LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset?select=lfw-deepfunneled) e [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/icon_zip.png), que trazem fotos de múltiplas personalidades. O primeiro dataset é organizado em pastas cujo título corresponde ao nome da pessoa. O segundo, diferentemente, não traz o nome das pessoas, apenas um número de imagem, e.g., 000467 e 012457. As identidades relativas a cada uma das fotos estão em um arquivo .txt auxiliar, assim como a localização localização das caixas delimitadoras (bounding boxes) para cada uma das fotos.

## Resumo do programa ([facial v8.py](https://github.com/msccobra/DIO-BairesDev/blob/main/Reconhecimento%20Facial/facial%20v8.py))
Este programa analisa imagens de rostos para identificar e recortar faces automaticamente. Ele trabalha com arquivos de imagem (como JPG) e utiliza anotações de texto para saber onde estão os rostos e quem são as pessoas nas fotos.

### Principais funcionalidades

- O usuário pode escolher o dataset usado para o reconhecimento facial e o número de identidades distintas a serem reconhecidas (serão sorteadas aleatoriamente). 
- Lê imagens de um diretório e suas anotações (arquivos .txt).
- Detecta e recorta rostos nas imagens.
- Salva os rostos recortados em pastas organizadas por identidade (nominal ou numérica, a depender do dataset usado).
- Prepara os dados para serem usados em modelos de reconhecimento facial.

### Tipos de arquivos analisados

- Imagens (.jpg)
- Arquivos de anotações (.txt) com informações de localização dos rostos e identidades
### Saídas do programa

- Um relatório completo no terminal com as identidades e a precisão na identificação, como o exemplo abaixo, copiado diretamente do terminal:

[RESULTADO FINAL] 70/70 acertos (100.00%)

=== Relatório por identidade ===

  Vladimir_Putin: 14/14 (100.00%)
  
  Spencer_Abraham: 14/14 (100.00%)
  
  Jacques_Chirac: 14/14 (100.00%)
  
  Nancy_Pelosi: 14/14 (100.00%)
  
  Angelina_Jolie: 14/14 (100.00%)

[DONE] Visualizações salvas em: F:\Documentos\outputs_vis

[DONE] Log CSV salvo em: F:\Documentos\outputs_vis\bboxes_log.csv

- Um arquivo de log .csv, com os dados provenientes da execução do programa
- Imagens de rostos recortados, salvos em pastas separadas por pessoa
- Um conjunto de dados pronto para treinar modelos de reconhecimento facial
