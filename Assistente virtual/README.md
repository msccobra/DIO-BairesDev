# Sistema de assistente virtual

O desafio da vez consistia em criar um [assistente virtual](https://github.com/msccobra/DIO-BairesDev/blob/main/Assistente%20virtual/DIO%20assistente.py) simples em Python. Para tal, foram usadas algumas bibliotecas, como a [gTTS (Google text-to-speech)](https://gtts.readthedocs.io/en/latest/index.html), que executa funções de síntese de voz de acordo com as entradas do usuário e a [Speech Recognition](https://pypi.org/project/SpeechRecognition/), que transforma entradas de voz em texto. Com o uso dessas duas ferramentas e mais algumas bibliotecas auxiliares, como a playsound, que foi utilizada para execução dos áudios, foi possível a criação de um assistente virtual simples por comando de voz com algumas funções, como: pesquisas no Google, pesquisas no YouTube, abra o Deezer, abra o Steam, que horas são? etc.

Para alguns dos inputs de voz do usuário, como no caso do YouTube e do Google, existe uma resposta do sistema perguntando qual a pesdquisa a ser feita, para que então as páginas sejam abertas. Em caso de o usuário pedir por um comando fora das opções dispoíveis, será retornada um mensagem de erro, tanto escrita como por áudio: "Comando não reconhecido".
Todas as opções disponíveis (que devem ser chamadas literalmente, como na lista), por ora, são:
- Abra o YouTube
- Abra o Google
- Abra o Deezer
- Abra o Steam
- Que horas são?
- Abra o Gmail

É um programa compacto, com menos de 100 linhas de código, que pode ser expandido para chamadas menos literais e mais opções de tarefas, além de uma UI de fato, mas o objetivo aqui, que era a demonstração da funcionalidade foi alcançado com êxito.
