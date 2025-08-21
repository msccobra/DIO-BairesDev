# Criando um sistema de reconhecimento facial

O desafio da vez foi a criação de um sistema de reconhecimento facial do zero, literalmente: "O objetivo principal deste projeto é trabalhar com as bibliotecas e frameworks estudados e analisados em nossas aulas. Neste sentido, a proposta padrão envolve um sistema de detecção e reconhecimento de faces, utilizando o framework TensorFlow em conjuntos com as bibliotecas que o projetista julgue necessárias, de forma ilimitada.".

Dessa maneira, foram desenvolvidos alguns programas de reconhecimento, os dois melhores esão disponíveis para download, link1 e link2. Nessa jornada, muitos problemas foram encontrados no caminho, especialmente a quetsão do estouro de RAM, pois as imagens estavam ficando armazenadas e não estavam sendo liberadas. Em um dataset como o CelebA, que tem cerca de 200k imagens, após cerca de 9,5k, meus 32 GB estouravam. Apesar de tudo, o programa rodava muito rápido, uma vitória inútil. Esses estouros eram causados pela maneira como o pacote MTCNN, que estava sendo usado, funciona.


