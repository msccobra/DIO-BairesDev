## Classificador e Sistema de Recomendação por Imagens de Moda
Este projeto utiliza Deep Learning para classificar imagens de produtos de moda (como relógios, camisetas, sapatos, etc.) e recomendar itens visualmente semelhantes. O objetivo é criar um sistema capaz de sugerir produtos relacionados com base na aparência física, facilitando buscas e recomendações em lojas online.

O programa começa carregando e preparando um dataset de imagens de moda, realizando a divisão dos dados em conjuntos de treino, validação e teste. Ele utiliza o modelo InceptionV3 pré-treinado como base, adicionando uma nova "cabeça" de classificação para adaptar o modelo às classes do seu dataset. O treinamento é feito em duas etapas: primeiro apenas a nova cabeça é treinada, depois parte do backbone é liberada para fine-tuning, melhorando a capacidade de reconhecimento visual.

Após o treinamento, o modelo é salvo no formato recomendado pelo Keras. O programa também gera relatórios de desempenho, como matriz de confusão e classification report, para avaliar a qualidade das previsões em cada classe.

Para o sistema de recomendação, o programa extrai "embeddings" das imagens do conjunto de teste usando a penúltima camada da rede neural. Esses embeddings representam características visuais aprendidas pelo modelo. Para cada classe, uma imagem é selecionada e o sistema busca as quatro imagens mais semelhantes, calculando a distância de similaridade entre os embeddings.

As recomendações são exibidas visualmente, mostrando a imagem consultada e as quatro sugestões mais próximas, facilitando a análise do funcionamento do sistema. O código está organizado para facilitar adaptações para outros datasets ou classes de produtos.

Este projeto é indicado para quem está começando com aprendizado profundo e quer entender como aplicar redes neurais em classificação de imagens e sistemas de recomendação visual. Basta seguir as instruções no código, ajustar os caminhos do dataset e executar o script para treinar e testar o modelo.
