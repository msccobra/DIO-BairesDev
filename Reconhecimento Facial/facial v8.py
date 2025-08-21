import os
import random
import shutil
import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
import tensorflow as tf

# ==========================
# CONFIGURAÇÕES
# ==========================
# Datasets
LFW_DIR = r"F:\Documentos\lfw-deepfunneled\lfw-deepfunneled"
IMG_DIR = r"F:\Documentos\celeba\Img\img_align_celeba\img_align_celeba"
ID_FILE = r"F:\Documentos\celeba\Anno\identity_CelebA.txt"
TEST_SUBSET_DIR = r"F:\Documentos\celeba\test_subset"

# Modelos
YOLO_FACE_WEIGHTS = r"F:\Downloads\yolov8s-face-lindevs.pt"
FACE_EMBEDDING_MODEL = r"F:\Downloads\arcfaceresnet100-8.onnx"

# Embeddings
EMBED_INPUT_SIZE = (112, 112)   # ArcFace
USE_COSINE = True               # cosine similarity
THRESHOLD = 0.35                # ajuste conforme validação

# Visualização
SAVE_VIS = True
VIS_DIR = r"F:\Documentos\outputs_vis"

# ==========================
# Detector YOLOv8 (faces)
# ==========================
class YOLOFaceDetector:
    def __init__(self, weights_path, conf=0.5):
        self.model = YOLO(weights_path)
        self.conf = conf

    def detect_all(self, img_bgr):
        res = self.model(img_bgr, conf=self.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy().astype(float)
        return [(tuple(b), float(c)) for b, c in zip(boxes, confs)]

    def detect_one_with_conf(self, img_bgr):
        dets = self.detect_all(img_bgr)
        if not dets:
            return None, None
        areas = [ (b[0][2]-b[0][0])*(b[0][3]-b[0][1]) for b in dets ]
        i = int(np.argmax(areas))
        return dets[i][0], dets[i][1]

    def detect_one(self, img_bgr):
        box, _ = self.detect_one_with_conf(img_bgr)
        return box

# ==========================
# Loaders de embedder (TF/ONNX)
# ==========================
def load_tf_embedder(model_path):
    if os.path.isdir(model_path):
        model = tf.saved_model.load(model_path)
        if "serving_default" in model.signatures:
            infer = model.signatures["serving_default"]
            def embed_fn(x):
                out = infer(x)
                first_key = list(out.keys())[0]
                return out[first_key]
            return embed_fn
        else:
            def embed_fn(x):
                return model(x)
            return embed_fn
    else:
        keras_model = tf.keras.models.load_model(model_path, compile=False)
        def embed_fn(x):
            return keras_model(x, training=False)
        return embed_fn

def load_onnx_embedder(model_path, providers=None):
    import onnxruntime as ort
    if providers is None:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    def embed_fn(batch_bhwc):
        if isinstance(batch_bhwc, tf.Tensor):
            batch_bhwc = batch_bhwc.numpy()
        x = np.transpose(batch_bhwc.astype(np.float32), (0, 3, 1, 2))  # NHWC -> NCHW
        out = sess.run(None, {input_name: x})[0]  # [B, D]
        return out
    return embed_fn

def load_embedder(model_path):
    ext = os.path.splitext(model_path)[1].lower()
    if ext == '.onnx':
        return load_onnx_embedder(model_path)
    tf_fn = load_tf_embedder(model_path)
    def wrapper(batch_bhwc):
        if isinstance(batch_bhwc, np.ndarray):
            batch_bhwc = tf.convert_to_tensor(batch_bhwc, dtype=tf.float32)
        return tf_fn(batch_bhwc).numpy()
    return wrapper

# ==========================
# Utils de pré-processo e visualização
# ==========================
def preprocess_face_for_embedder(face_bgr, target_size=(112, 112)):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    face_rgb = face_rgb.astype(np.float32)
    face_rgb = (face_rgb / 127.5) - 1.0  # [-1,1]
    return face_rgb

def crop_with_margin(img, box, margin=0.2):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    dx = int(bw * margin)
    dy = int(bh * margin)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(w, x2 + dx)
    ny2 = min(h, y2 + dy)
    return img[ny1:ny2, nx1:nx2]

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))

def l2_distance(a, b):
    return float(np.linalg.norm(a - b))

def save_annotated(img_bgr, box, out_path, label=None, color=(0,255,0), thickness=2):
    if box is None:
        return
    x1,y1,x2,y2 = map(int, box)
    vis = img_bgr.copy()
    cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(0, y1 - th - 6)
        cv2.rectangle(vis, (x1, y0), (x1 + tw + 6, y0 + th + 6), color, -1)
        cv2.putText(vis, label, (x1 + 3, y0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, vis)

# ==========================
# SUBSETS
# ==========================
def create_test_subset_celeba(num_identities=100, min_images_per_identity=5, copy_images=True):
    id_df = pd.read_csv(ID_FILE, delim_whitespace=True, header=None, names=["image_id", "identity"])
    grouped = id_df.groupby("identity")["image_id"].apply(list)
    valid_identities = [person for person, imgs in grouped.items() if len(imgs) >= min_images_per_identity]
    print(f"[INFO] CelebA - Identidades com >= {min_images_per_identity} imagens: {len(valid_identities)}")
    if len(valid_identities) == 0:
        print("[ERROR] Nenhuma identidade com imagens suficientes!")
        return None
    selected_identities = random.sample(valid_identities, min(num_identities, len(valid_identities)))
    print(f"[INFO] Selecionadas {len(selected_identities)} identidades")
    if copy_images:
        if os.path.exists(TEST_SUBSET_DIR):
            shutil.rmtree(TEST_SUBSET_DIR)
        os.makedirs(TEST_SUBSET_DIR, exist_ok=True)
    subset_info = {}
    for person in selected_identities:
        imgs = grouped[person]
        chosen_imgs = random.sample(imgs, min_images_per_identity)
        subset_info[person] = chosen_imgs
        if copy_images:
            person_dir = os.path.join(TEST_SUBSET_DIR, str(person))
            os.makedirs(person_dir, exist_ok=True)
            for f in chosen_imgs:
                src = os.path.join(IMG_DIR, f)
                dst = os.path.join(person_dir, f)
                shutil.copy(src, dst)
    print(f"[DONE] Subconjunto CelebA salvo em: {TEST_SUBSET_DIR}")
    return TEST_SUBSET_DIR

def create_test_subset_lfw(num_identities=100, min_images_per_identity=5):
    people = [p for p in os.listdir(LFW_DIR) if os.path.isdir(os.path.join(LFW_DIR, p))]
    valid_people = [p for p in people if len(os.listdir(os.path.join(LFW_DIR, p))) >= min_images_per_identity]
    print(f"[INFO] LFW - Pessoas com >= {min_images_per_identity} imagens: {len(valid_people)}")
    if len(valid_people) == 0:
        print("[ERROR] Nenhuma identidade com imagens suficientes!")
        return None
    selected = random.sample(valid_people, min(num_identities, len(valid_people)))
    return selected

# ==========================
# TESTE (YOLOv8 + Embeddings TF/ONNX) com visualização
# ==========================
def test_face_recognition_tf_yolo(dataset_dir, detector, embedder_fn,
                                  num_identities=100, max_images_per_person=5,
                                  is_lfw=False, selected_people=None,
                                  save_vis=False, vis_dir="outputs"):
    if not is_lfw:
        people = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        test_people = random.sample(people, min(num_identities, len(people)))
    else:
        test_people = selected_people

    total_tests = 0
    correct = 0
    report = {}
    log_rows = []

    for person in test_people:
        person_dir = os.path.join(dataset_dir, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < 2:
            continue

        # Referência
        ref_path = os.path.join(person_dir, images[0])
        ref_bgr = cv2.imread(ref_path)
        if ref_bgr is None:
            continue
        ref_box, ref_conf = detector.detect_one_with_conf(ref_bgr)
        if ref_box is None:
            continue

        if save_vis:
            out_ref = os.path.join(vis_dir, str(person), f"_REF_{os.path.basename(ref_path)}")
            save_annotated(ref_bgr, ref_box, out_ref, label=f"ref conf={ref_conf:.2f}")

        ref_crop = crop_with_margin(ref_bgr, ref_box, margin=0.2)
        ref_input = preprocess_face_for_embedder(ref_crop, target_size=EMBED_INPUT_SIZE)
        ref_emb = embedder_fn(ref_input[None, ...]).squeeze()
        if USE_COSINE:
            ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-8)

        if save_vis:
            log_rows.append({
                "person": person, "image": os.path.basename(ref_path),
                "x1": ref_box[0], "y1": ref_box[1], "x2": ref_box[2], "y2": ref_box[3],
                "conf": ref_conf, "is_ref": True, "score": None, "match": None
            })

        identity_total = 0
        identity_correct = 0

        # Testes
        for test_img_file in images[1:max_images_per_person]:
            test_path = os.path.join(person_dir, test_img_file)
            test_bgr = cv2.imread(test_path)
            if test_bgr is None:
                continue
            test_box, test_conf = detector.detect_one_with_conf(test_bgr)
            if test_box is None:
                continue

            test_crop = crop_with_margin(test_bgr, test_box, margin=0.2)
            test_input = preprocess_face_for_embedder(test_crop, target_size=EMBED_INPUT_SIZE)
            test_emb = embedder_fn(test_input[None, ...]).squeeze()

            if USE_COSINE:
                test_emb = test_emb / (np.linalg.norm(test_emb) + 1e-8)
                score = cosine_similarity(ref_emb, test_emb)
                is_same = score >= THRESHOLD
            else:
                score = l2_distance(ref_emb, test_emb)
                is_same = score <= THRESHOLD

            identity_total += 1
            total_tests += 1
            if is_same:
                identity_correct += 1
                correct += 1

            if save_vis:
                color = (0, 200, 0) if is_same else (0, 0, 255)
                out_test = os.path.join(vis_dir, str(person), os.path.basename(test_path))
                lbl = f"sim={score:.2f} {'OK' if is_same else 'NO'} conf={test_conf:.2f}"
                save_annotated(test_bgr, test_box, out_test, label=lbl, color=color)
                log_rows.append({
                    "person": person, "image": os.path.basename(test_path),
                    "x1": test_box[0], "y1": test_box[1], "x2": test_box[2], "y2": test_box[3],
                    "conf": test_conf, "is_ref": False, "score": float(score), "match": bool(is_same)
                })

        if identity_total > 0:
            acc = identity_correct / identity_total * 100
            report[person] = (identity_correct, identity_total, acc)

    if total_tests > 0:
        acc = correct / total_tests * 100
        print(f"\n[RESULTADO FINAL] {correct}/{total_tests} acertos ({acc:.2f}%)")
        print("=== Relatório por identidade ===")
        for person, stats in report.items():
            c, t, p = stats
            print(f"  {person}: {c}/{t} ({p:.2f}%)")
    else:
        print("[WARNING] Nenhum teste válido realizado")

    if save_vis and log_rows:
        os.makedirs(vis_dir, exist_ok=True)
        csv_path = os.path.join(vis_dir, "bboxes_log.csv")
        pd.DataFrame(log_rows).to_csv(csv_path, index=False)
        print(f"[DONE] Visualizações salvas em: {vis_dir}")
        print(f"[DONE] Log CSV salvo em: {csv_path}")

# ==========================
# MAIN
# ==========================
def main():
    print("=== PIPELINE DE RECONHECIMENTO (YOLOv8 + Embeddings TF/ONNX) ===\n")

    # GPU opcional no TF
    try:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] Não foi possível ajustar memory_growth do TF: {e}")

    # Checagens
    if not os.path.exists(YOLO_FACE_WEIGHTS):
        print(f"[WARN] Pesos YOLO não encontrados em: {YOLO_FACE_WEIGHTS}")
    if not (os.path.exists(FACE_EMBEDDING_MODEL) or os.path.isdir(FACE_EMBEDDING_MODEL)):
        print(f"[WARN] Modelo de embeddings não encontrado em: {FACE_EMBEDDING_MODEL}")

    print("[INFO] Carregando detector YOLOv8...")
    detector = YOLOFaceDetector(YOLO_FACE_WEIGHTS, conf=0.5)

    print("[INFO] Carregando modelo de embeddings (ONNX/TF)...")
    embedder_fn = load_embedder(FACE_EMBEDDING_MODEL)

    dataset_choice = input("Qual dataset deseja usar? (1) CelebA (2) LFW : ").strip()
    try:
        num_identities = int(input("Quantas identidades aleatórias? (default=100): ") or 100)
        min_images = int(input("Quantas imagens mínimas por identidade? (default=5): ") or 5)
    except Exception:
        num_identities, min_images = 100, 5

    print(f"[CONFIG] Similaridade: {'cosine' if USE_COSINE else 'L2'} | Limite: {THRESHOLD} | Input: {EMBED_INPUT_SIZE}")
    print(f"[CONFIG] Visualizações: {SAVE_VIS} | Pasta: {VIS_DIR}")

    if dataset_choice == "1":
        dataset_dir = create_test_subset_celeba(
            num_identities=num_identities,
            min_images_per_identity=min_images,
            copy_images=True
        )
        if dataset_dir:
            test_face_recognition_tf_yolo(
                dataset_dir,
                detector,
                embedder_fn,
                num_identities=num_identities,
                max_images_per_person=min_images,
                is_lfw=False,
                save_vis=SAVE_VIS,
                vis_dir=VIS_DIR
            )

    elif dataset_choice == "2":
        selected = create_test_subset_lfw(
            num_identities=num_identities,
            min_images_per_identity=min_images
        )
        if selected:
            print(f"[INFO] Rodando teste em {len(selected)} pessoas (LFW)")
            for person in selected[:5]:
                print(" -", person)
            test_face_recognition_tf_yolo(
                LFW_DIR,
                detector,
                embedder_fn,
                num_identities=num_identities,
                max_images_per_person=min_images,
                is_lfw=True,
                selected_people=selected,
                save_vis=SAVE_VIS,
                vis_dir=VIS_DIR
            )
    else:
        print("[ERROR] Escolha inválida! Digite 1 (CelebA) ou 2 (LFW).")

if __name__ == "__main__":
    main()