import os
import pandas as pd
import random
import shutil
import face_recognition


# Caminhos dos arquivos dos datasets (ajuste conforme necessário, mesmo links podem ser usados)

# O dataset LFW (Labelled Faces in the Wild) contém fotos de celebridades com identificação nominal (uma por celebridade)
# >>> Caminho do dataset LFW

LFW_DIR = r"F:\Documentos\lfw-deepfunneled\lfw-deepfunneled"

# O dataset CelebA contém fotos de celebridades com identificações em forma numérica
# >>> Diretórios CelebA

IMG_DIR = r"F:\Documentos\celeba\Img\img_align_celeba\img_align_celeba"
ID_FILE = r"F:\Documentos\celeba\Anno\identity_CelebA.txt"
TEST_SUBSET_DIR = r"F:\Documentos\celeba\test_subset"

# Criar subconjunto de teste para CelebA
# Aqui você pode ajustar o número de identidades e imagens mínimas por identidade, o padrão é 100 identidades com pelo menos 5 imagens cada
# Ao rodar o programa, o usuário poderá escolher o número de identidades e imagens mínimas

def create_test_subset_celeba(num_identities=100, min_images_per_identity=5, copy_images=True):
    id_df = pd.read_csv(ID_FILE, delim_whitespace=True, header=None, names=["image_id", "identity"])
    grouped = id_df.groupby("identity")["image_id"].apply(list)

    # Identidades com >= M imagens
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


# Criar subconjunto de teste para LFW
# De maneira similar, os mesmos valores padrão de número de identidades e imagens mínimas por identidade são usados
# Ao rodar o programa, o usuário poderá escolher o número de identidades e imagens mínimas

def create_test_subset_lfw(num_identities=100, min_images_per_identity=5):
    people = [p for p in os.listdir(LFW_DIR) if os.path.isdir(os.path.join(LFW_DIR, p))]
    # pessoas com >= M imagens
    valid_people = [p for p in people if len(os.listdir(os.path.join(LFW_DIR, p))) >= min_images_per_identity]
    print(f"[INFO] LFW - Pessoas com >= {min_images_per_identity} imagens: {len(valid_people)}")

    if len(valid_people) == 0:
        print("[ERROR] Nenhuma identidade com imagens suficientes!")
        return None
    # Seleciona aleatoriamente N pessoas no dataset
    selected = random.sample(valid_people, min(num_identities, len(valid_people)))
    return selected

# Teste de reconhecimento (funciona tanto p/ CelebA quanto p/ LFW)
# A função retorna o total de acertos (% geral), além de um relatório por identidade (numérica ou nominal, a depender do dataset)

def test_face_recognition(dataset_dir, num_identities=100, max_images_per_person=5, is_lfw=False, selected_people=None):
    """
    Testa reconhecimento facial nos subconjuntos LFW ou CelebA
    """
    # Para CelebA, pega diretórios de identidades
    if not is_lfw:
        people = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        test_people = random.sample(people, min(num_identities, len(people)))
    else:
        # Para LFW, já recebemos as identities selecionadas
        test_people = selected_people

    total_tests = 0
    correct = 0
    report = {}

    for person in test_people:
        person_dir = os.path.join(dataset_dir, person)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg','.png'))]
        if len(images) < 2:
            continue

        # referência
        ref_img = face_recognition.load_image_file(os.path.join(person_dir, images[0]))
        ref_enc = face_recognition.face_encodings(ref_img)
        if len(ref_enc) == 0:
            continue
        ref_enc = ref_enc[0]

        identity_total = 0
        identity_correct = 0

        for test_img_file in images[1:max_images_per_person]:
            test_img = face_recognition.load_image_file(os.path.join(person_dir, test_img_file))
            encs = face_recognition.face_encodings(test_img)
            if len(encs) == 0:
                continue
            test_enc = encs[0]

            result = face_recognition.compare_faces([ref_enc], test_enc, tolerance=0.6)[0]
            identity_total += 1
            total_tests += 1
            if result:
                identity_correct += 1
                correct += 1

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

# Main

def main():
    print("=== PIPELINE DE RECONHECIMENTO COM CELEBA OU LFW ===\n")
    dataset_choice = input("Qual dataset deseja usar? (1) CelebA (2) LFW : ").strip()

    try:
        num_identities = int(input("Quantas identidades aleatórias? (default=100): ") or 100)
        min_images = int(input("Quantas imagens mínimas por identidade? (default=5): ") or 5)
    except:
        num_identities, min_images = 100, 5

    if dataset_choice == "1":
        dataset_dir = create_test_subset_celeba(num_identities=num_identities, min_images_per_identity=min_images, copy_images=True)
        if dataset_dir:
            test_face_recognition(dataset_dir, num_identities=num_identities, max_images_per_person=min_images, is_lfw=False)
    
    elif dataset_choice == "2":
        selected = create_test_subset_lfw(num_identities=num_identities, min_images_per_identity=min_images)
        if selected:
            print(f"[INFO] Rodando teste em {len(selected)} pessoas (LFW)")
            for person in selected[:5]:
                print(" -", person)
            test_face_recognition(LFW_DIR, num_identities=num_identities, max_images_per_person=min_images, is_lfw=True, selected_people=selected)

    else:
        print("[ERROR] Escolha inválida! Digite 1 (CelebA) ou 2 (LFW).")

if __name__ == "__main__":
    main()