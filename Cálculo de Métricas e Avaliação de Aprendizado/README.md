# Cálculo de Métricas de Avaliação de Aprendizado

O desafio da vez consiste na exploração das métricas de desempenho de modelos de classificação. A métrica mais comum usada nesses casos é a matriz de confusão, assim como índices de desmpenho derivados dos dados presentes nela. A matriz de confusão é uma tabela na qual estão os dados comparativos obtidos pelo modelo de classificação e as observações reais (dados reais), classificando-os, de maneira simplificada para um sistema binário (de duas classes), em verdadeiro positivo (VP), onde a observação real e o modelo concordam com a classificação positiva do objeto analisado, e.g., se analisados os marcadores de uma doença, ambos concordam que a doença existe, ou, o sistema classifica uma imagem como sendo um gato e é uma gato; verdadeiro negativo (VN) ocorre quando a predição do modelo e a observação real estão em consonância em negar aquilo que seria considerado o controle positivo. Neste caso, o exemplo seria o oposto do anterior: o programa classifica o paciente como não portador de uma doença (basdeado em seus marcadores) e ele não está doente, ou, o sistema classifica o animal como não sendo um gato, e ele não é. Os dois casos seguintes, por outro lado, retratam erros do sistema de classificação: falso positivo (FP) e falso negativo (FN). O primeiro refere-se a quando o modelo classifica a entrada como sendo similar ao controle positivo, mas a observação real indica o contrário, e.g., o sistema classifica uma foto de um animal como sendo um gato, e ele não é. Já o falso negativo é o oposto do caso anterior. É quando o modelo classifica a entrada como sendo de uma classe diferente do controle positivo e ela pertence à mesma classe do controle positivo, e.g., o modelo classifica uma foto como não sendo de um gato, e ela é.

Um exemplo de matriz de confusão está abaixo:


|                | Predição: Negativo | Predição: Positivo |
|----------------|-------------------|-------------------|
| **Real: Negativo** | VN              | FP                |
| **Real: Positivo** | FN                | VP             |

Dos valores da matriz de confusão, podemos obter algumas métricas auxiliares de avaliação de desempenho do modelo, são elas:

Precisão (P): $\frac{VP}{VP+FP}$

Acurácia (A): $\frac{VP+VN}{Pop. Total}$

Especificidade (E): $\frac{VN}{VN+FP}$

Sensibilidade (S): $\frac{VP}{VP+FN}$

F-Score: $2x\frac{PxS}{P+S}$

Como exemplo de uma aplicação real da matriz de confusão, foi feita uma atualização do programa [Transfer v2](https://github.com/msccobra/DIO-BairesDev/blob/main/Transfer%20Learning/Transfer%20v2.py), de classificação de imagens, para que fossem incluídas as métricas auxiliares de desempenho aos dados da matriz de confusão já existente. O programa [Transfer v3](https://github.com/msccobra/DIO-BairesDev/blob/main/C%C3%A1lculo%20de%20M%C3%A9tricas%20e%20Avalia%C3%A7%C3%A3o%20de%20Aprendizado/Transfer%20v3.py) conta com o cálculo de todas as métricas auxiliares citadas acima. As seguintes linhas foram adicionadas ao programa para tal:
```
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
```
Enfim, o programa Transfer v3, usando como banco de dados [Kaggle cats and dogs](https://download.microsoft.com/download/3/e/1/3e1c3f21-ecdb-4869-8368-6deba77b919f/kagglecatsanddogs_5340.zip) para treinamento/validação/teste, como explicado no projeto de [Transfer Learning](https://github.com/msccobra/DIO-BairesDev/tree/main/Transfer%20Learning), resultou na seguinte matriz de confusão e em suas respectivas métricas:

Matriz de confusão:

|                | Predição: Negativo | Predição: Positivo |
|----------------|-------------------|-------------------|
| **Real: Negativo** | 2477             | 23               |
| **Real: Positivo** | 12              | 2488             |

**Obs: os valores negativos referem-se às fotos de cães e os positivos a fotos de gatos.**

Precisão: 0.9930

Sensibilidade: 0.9952

Especificidade: 0.9908

Acurácia: 0.9930

F-score: 0.9941
