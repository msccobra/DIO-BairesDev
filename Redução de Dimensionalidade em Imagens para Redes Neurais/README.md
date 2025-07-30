# Redução da dimensionalidade em imagens

O objetivo desse projeto foi o desenvolvimento de um algoritmo para a redução de dimensionalidade em imagens, i.e., redução do número de cores mostradas. Neste caso, o objetivo da redução do número de cores é, primariamente, a redução do tamanho da imagem a ser armazenada. A [imagem original](https://github.com/msccobra/DIO-BairesDev/blob/main/Redu%C3%A7%C3%A3o%20de%20Dimensionalidade%20em%20Imagens%20para%20Redes%20Neurais/Crie%20um%20quadro%20surre.png) usada foi uma brincadeira que eu fiz em um projeto anterior da DIO, na qual eu escrevi um prompt bastante detalhado (com cada detalhe que eu queria na imagem) pedindo para a IA criar um quadro no estilo de Salvador Dalí. A mudança de tamanho na imagem armazenada original para a [imagem em escala de cinza](https://github.com/msccobra/DIO-BairesDev/blob/main/Redu%C3%A7%C3%A3o%20de%20Dimensionalidade%20em%20Imagens%20para%20Redes%20Neurais/Crie%20um%20quadro%20surre_grayscale_direct.png) foi de 2,88 MB para 0,98 MB. Essa redução no espaço de armazenamento é fundamental em casos nos quais existe um grande fluxo de informação, como em sistemas de reconhecimento facial, onde as cores não são relevantes no cumprimento das tarefas. Uma segunda etapa foi a transformação da imagem original em uma [imagem binária](https://github.com/msccobra/DIO-BairesDev/blob/main/Redu%C3%A7%C3%A3o%20de%20Dimensionalidade%20em%20Imagens%20para%20Redes%20Neurais/Crie%20um%20quadro_pb.png) (preto e branco). Esse tipo de transformação, além da redução do espaço de armazenamento requerido (a imagem P&B tem apenas 30 kB), pode resultar em imagens adequadas para a identificação de contornos, que pode ser importante em um algoritmo de reconehcimento de objetos. No caso da imagem usada por mim, o [resultado](https://github.com/msccobra/DIO-BairesDev/blob/main/Redu%C3%A7%C3%A3o%20de%20Dimensionalidade%20em%20Imagens%20para%20Redes%20Neurais/Crie%20um%20quadro_pb.png) não foi tão bom, embora alguma coisa do contorno do elefante alado possa ser identificada.

O grande desafio desse projeto foi o desenvolvimento de um programa que fizesse as tarefas acima sem a utilização de uma biblioteca Python para tal, como a PIL, que a transformação fosse programada do zero. Assim foi feito, apesar de constar a biblioteca PIL no programa, ela foi usada apenas para os processos que envolviam abrir e salvar os arquivos, não na manipulação dos mesmos. Esse desafio foi a prova de que, se há um trabalho já feito, uma biblioteca já consagrada pelo uso, não há necessidade de fazer todo o trabalho de novo. Algo que poderia ter sido resolvido em 5 linhas, foi resolvido em mais de 100, e com (bastante) ajuda do Google e das IA's.

## O programa de transformação da imagem para escala de cinza

O programa usado para transformação da imagem original em uma na escala de cinza está abaixo. Note que eu usei as pastas e arquivos de entradas e saídas locais. Isso pode ser facilmente modificado para links de entrada na nuvem. Preferi essa abordagem, pois tive que fazer um extenso debug do código, e usar armazenamento local é um fator a menos de erro. O programa carrega uma imagem, a modifica para o formato ppm, faz as transformações de cor e salva a imagem modificada em formato png.

Aqui estão os links para os programas de transformação para [escala de cinza](https://github.com/msccobra/DIO-BairesDev/blob/main/Redu%C3%A7%C3%A3o%20de%20Dimensionalidade%20em%20Imagens%20para%20Redes%20Neurais/Imagens%20gs.py) e [P&B](https://github.com/msccobra/DIO-BairesDev/blob/main/Redu%C3%A7%C3%A3o%20de%20Dimensionalidade%20em%20Imagens%20para%20Redes%20Neurais/Imagens%20pb.py).

```
from PIL import Image
from pathlib import Path
import os

# Foi usada a biblioteca PIL apenas para abrir e salvar imagens, não foram usados os comandos da mesma
# Foi feito o código para a conversão de uma imagem em cores para uma imagem em escala de cinza

# Caminhos (Substitua para o seu caso)
entrada = r'C:\Users\flawl\OneDrive\Imagens\Crie um quadro surre.png'
saida_ppm = r'C:\Temp\Crie um quadro surre.ppm'
saida_grayscale = r'C:\Temp\Crie um quadro surre_grayscale.ppm'
saida_png = r'C:\Temp\Crie um quadro surre_grayscale.png'

# Criar diretório C:\Temp se não existir (usado para debug, deixei no código final)
Path(r'C:\Temp').mkdir(exist_ok=True)

# Etapa 1: Verificar e converter PNG para PPM (gera P6)
# Essa primeira etapa é necessária para a conversão de uma imagem sem uso da biblioteca PIL (embora ela esteja sendo usada para abrir e salvar imagens)
# Os arquivos PPM são uma maneira de armazenar imagens em formato de texto ou binário
# O formato P6 é binário, enquanto P3 é texto
try:
    imagem = Image.open(entrada)
    width, height = imagem.size
    print(f"Dimensões do PNG: {width}x{height}")
    expected_pixels = width * height
    imagem.save(saida_ppm, format='PPM')
    print(f"Arquivo PPM salvo em: {saida_ppm}")
    # Verificar tamanho do arquivo
    ppm_size = os.path.getsize(saida_ppm)
    expected_size = expected_pixels * 3 + 50  # Aproximado, incluindo cabeçalho
    print(f"Tamanho do arquivo PPM P6: {ppm_size} bytes (esperado: ~{expected_size} bytes)")
except FileNotFoundError:
    print(f"Erro: Arquivo PNG '{entrada}' não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao abrir ou converter PNG para PPM: {e}")
    exit()

# Etapa 2: Validar o arquivo PPM P6 com PIL
try:
    ppm_image = Image.open(saida_ppm)
    ppm_width, ppm_height = ppm_image.size
    print(f"Dimensões do PPM P6 lido pelo PIL: {ppm_width}x{ppm_height}")
    if ppm_width != width or ppm_height != height:
        print(f"Erro: Dimensões do PPM P6 ({ppm_width}x{ppm_height}) não correspondem ao PNG ({width}x{height})")
        exit()
except Exception as e:
    print(f"Erro ao abrir PPM P6 com PIL: {e}")
    exit()

# Etapa 3: Converter PPM P6 para escala de cinza (salva como P3)
def ppm_to_grayscale(input_path, output_path):
    try:
        with open(input_path, 'rb') as f:
            # Ler header
            header = []
            # Ler P6
            while len(header) < 3:
                line = f.readline()
                # Ignorar comentários
                if line.startswith(b'#'):
                    continue
                header.append(line.strip().decode('ascii'))
                
            if header[0] != 'P6':
                raise ValueError(f"Apenas formato PPM P6 é suportado, encontrado: {header[0]}")

            width, height = map(int, header[1].split())
            max_val = int(header[2])
            print(f"Formato: P6, Dimensões: {width}x{height}, Valor máximo: {max_val}")

            # Posição atual do arquivo: dados de pixel começam aqui
            pixel_data = f.read()
            expected_pixels = width * height
            if len(pixel_data) != expected_pixels * 3:
                raise ValueError(f"Erro: Esperado {expected_pixels*3} bytes de pixel, mas lidos {len(pixel_data)}")
            
            # Processa pixels
            grayscale_pixels = []
            for i in range(0, len(pixel_data), 3):
                r, g, b = pixel_data[i:i+3]
                gray = int(0.299 * r + 0.587 * g + 0.114 * b)
                if gray > max_val: gray = max_val
                grayscale_pixels.append((gray, gray, gray))

            # Escrever arquivo PPM P3 (texto)
            with open(output_path, 'w') as f_out:
                f_out.write('P3\n')
                f_out.write(f'{width} {height}\n')
                f_out.write(f'{max_val}\n')
                line_length = 0
                for r, g, b in grayscale_pixels:
                    f_out.write(f'{r} {g} {b} ')
                    line_length += 1
                    if line_length >= 5 * 3:
                        f_out.write('\n')
                        line_length = 0
                if line_length > 0:
                    f_out.write('\n')
            print(f"Arquivo PPM em escala de cinza salvo em: {output_path}")

    except Exception as e:
        print(f"Erro ao converter para escala de cinza: {e}")
        exit()

ppm_to_grayscale(saida_ppm, saida_grayscale)

# Etapa 4: Reconverter PPM P3 para PNG
try:
    with open(saida_grayscale, 'r') as f:
        ppm_lines = f.readlines()
        data_lines = [line for line in ppm_lines[3:] if line.strip() and not line.startswith('#')]
    imagem_grayscale = Image.open(saida_grayscale)
    imagem_grayscale.save(saida_png, format='PNG')
    print(f"Arquivo PNG em escala de cinza salvo em: {saida_png}")
except FileNotFoundError:
    print(f"Erro: Arquivo PPM '{saida_grayscale}' não encontrado.")
    exit()
except Exception as e:
    print(f"Erro ao converter PPM para PNG: {e}")
    exit()
