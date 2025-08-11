# Criação de Uma Base de Dados e Treinamento da Rede YOLO

Este projeto consiste em criar um programa para detecção e rotulamento de objetos usando a rede Yolo. Para tal, foi desenvolvido um programa no qual os rótulos, no qual havia um arquivo .json correpondente a cada imagem, com uma breve descrição, é lido, mapeando cada imagem e a ligando aos seus respectivos rótulos. A descrição de cada imagem no arquivo .json está no formato:

{"license": 1,"file_name": "000000037777.jpg","coco_url": "http://images.cocodataset.org/val2017/000000037777.jpg", "height": 230,"width": 352,"date_captured": "2013-11-14 20:55:31", "flickr_url": "http://farm9.staticflickr.com/8429/7839199426_f6d48aa585_z.jpg", "id": 37777}, {"image_id": 37777,"id": 597185,"caption": "The dining table near the kitchen has a bowl of fruit on it."}. 

Após isso as imagens e suas respectivas anotações são divididas em dois grupos, "train" e "val", grupos de treino e validação do modelo. O programa então mapeia cada uma das imagens e define as coordenadas das caixas de identificação dos objetos no seguinte formato, onde cada linha corersponde a um objeto, e os primeiros números corerspondem à classe do mesmo, dentre as 80 classes definidas pelo dataset COCO:

- 50 0.157586 0.533708 0.308422 0.424708
- 50 0.304344 0.065479 0.194594 0.127500
- 48 0.802312 0.812729 0.245875 0.374542
- 53 0.504086 0.696625 0.529516 0.606750
- 53 0.910953 0.660823 0.170219 0.642687
- 66 0.837078 0.207865 0.325844 0.415729
- 57 0.905273 0.677094 0.189453 0.580729

Após essa definição, o script recarrega as labels geradas para desenhar as caixas de identificação e nomes de classe diretamente sobre as imagens usando a ferramenta Annotator. As imagens classificadas são, então, salvas em uma pasta de saída, permitindo visualização imediata dos resultados e conferência da qualidade das anotações.Um exemplo de imagem classificada está no link a seguir: [Imagem](https://github.com/msccobra/DIO-BairesDev/blob/main/Cria%C3%A7%C3%A3o%20de%20Uma%20Base%20de%20Dados%20e%20Treinamento%20da%20Rede%20YOLO/annotated_000000009914.jpg)
