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

