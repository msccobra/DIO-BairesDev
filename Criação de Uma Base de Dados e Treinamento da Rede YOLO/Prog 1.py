import json
import os
from ultralytics import YOLO
import random
from pathlib import Path
import argparse
import cv2
from ultralytics.utils.plotting import Annotator

model = YOLO("yolov8n.pt")

# Paths
coco_json = r"C:\Users\flawl\Documentos\Projeto DIO\Data\Annotations\instances_val2017.json"
images_dir = r"C:\Users\flawl\Documentos\Projeto DIO\Data\Images\val2017"
labels_dir = r"C:\Users\flawl\Documentos\Projeto DIO\Labels\val"
output_dir = r"C:\Users\flawl\Documentos\Projeto DIO\Annotated_Images"
os.makedirs(labels_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Converte anotações COCO para YOLO e faz split em train/val"
    )
    parser.add_argument(
        "--coco_json", type=Path, required=True,
        help="Caminho para o arquivo JSON COCO (ex: Data/annotations/instances.json)"
    )
    parser.add_argument(
        "--img_dir", type=Path, required=True,
        help="Pasta com todas as imagens (ex: Data/images)"
    )
    parser.add_argument(
        "--labels_dir", type=Path, required=True,
        help="Pasta onde serão salvas as labels (ex: Labels)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Porcentagem de imagens para treino (0.0–1.0)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Semente para divisão aleatória"
    )
    return parser.parse_args()


def load_coco_annotations(coco_json):
    with open(coco_json, "r", encoding="utf-8") as f:
        return json.load(f)

def split_images(image_list, train_ratio, seed):
    random.seed(seed)
    shuffled = image_list.copy()
    random.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    return set(shuffled[:split]), set(shuffled[split:])

def main():
    #args = parse_args()

    coco = load_coco_annotations(coco_json)
    images = coco["images"]
    annotations = coco["annotations"]

    # Map image_id → image info
    img_info = { img["id"]: img for img in images }

    # Group annotations por image_id
    anns_by_img = {}
    for ann in annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Coleta lista de arquivos de imagem existentes
    all_files = {p.name for p in Path(images_dir).iterdir() if p.is_file()}
    
    # Divide arquivos entre train e val
    train_ratio = 0.8
    seed = 42
    train_files, val_files = split_images(
        [info["file_name"] for info in images if info["file_name"] in all_files],
        train_ratio, seed
    )

    # Cria pastas para labels
    train_lbl_dir = Path(labels_dir) / "train"
    val_lbl_dir   = Path(labels_dir) / "val"
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Função para processar subset
    def process_subset(filenames, out_dir):
        for fname in filenames:
            info = img_info[next(
                img_id for img_id, i in img_info.items() if i["file_name"] == fname
            )]
            H, W = info["height"], info["width"]
            anns = anns_by_img.get(info["id"], [])
            if not anns:
                continue  # pula imagem sem anotações

            out_path = out_dir / fname.replace(".jpg", ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for ann in anns:
                    cls_id = ann["category_id"] - 1
                    x, y, w, h = ann["bbox"]
                    x_c = (x + w/2) / W
                    y_c = (y + h/2) / H
                    f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w/W:.6f} {h/H:.6f}\n")

    # Gera arquivos .txt
    process_subset(train_files, train_lbl_dir)
    process_subset(val_files,   val_lbl_dir)

    print("Conversão concluída!")
    print(f"  • Treino: {len(list(train_lbl_dir.iterdir()))} labels")
    print(f"  • Validação: {len(list(val_lbl_dir.iterdir()))} labels")

if __name__ == "__main__":
    main()

# Carregar JSON do COCO para obter nomes das classes
with open(coco_json, "r", encoding="utf-8") as f:
    coco = json.load(f)
id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}

# Iterar sobre as imagens
for img_file in os.listdir(images_dir):
    if img_file.endswith(".jpg"):  # Assumindo que as imagens são .jpg
        img_path = os.path.join(images_dir, img_file)
        label_file = img_file.replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, label_file)

        # Pular se não houver arquivo de rótulo
        if not os.path.exists(label_path):
            continue

        # Carregar imagem
        im0 = cv2.imread(img_path)
        if im0 is None:
            continue

        # Criar anotador
        annotator = Annotator(im0)

        # Ler rótulos
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_c_norm = float(parts[1])
                y_c_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Obter dimensões da imagem
                h_img, w_img, _ = im0.shape

                # Converter coordenadas normalizadas para pixels
                x_c_pixel = x_c_norm * w_img
                y_c_pixel = y_c_norm * h_img
                w_pixel = w_norm * w_img
                h_pixel = h_norm * h_img

                # Calcular cantos da caixa delimitadora
                x1 = int(x_c_pixel - w_pixel / 2)
                y1 = int(y_c_pixel - h_pixel / 2)
                x2 = int(x_c_pixel + w_pixel / 2)
                y2 = int(y_c_pixel + h_pixel / 2)

                # Obter nome da classe
                category_id = cls_id + 1  # cls_id = category_id - 1
                class_name = id_to_name.get(category_id, f"unknown_{category_id}")

                # Desenhar caixa e rótulo
                annotator.box_label([x1, y1, x2, y2], class_name)

        # Obter imagem anotada
        annotated_im = annotator.result()

        # Salvar imagem anotada
        cv2.imwrite(os.path.join(output_dir, f"annotated_{img_file}"), annotated_im)

print("Visualização concluída! Verifique o diretório de saída para imagens anotadas.")


