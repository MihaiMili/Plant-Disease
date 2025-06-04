import os
import cv2
import random
import sys
import numpy as np
from ultralytics import YOLO

# === Configurare căi ===
model_path = r'C:\Users\mihai\runs\detect\train14\weights\best.pt'
image_folder = r'C:\Users\mihai\Downloads\cartof\Poze'
output_folder = r'C:\Users\mihai\Downloads\cartof\rezultate'
os.makedirs(output_folder, exist_ok=True)

# === Încarcă modelul ===
model = YOLO(model_path)

# === Obține rezoluția ecranului ===
def get_screen_resolution():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    root.destroy()
    return width, height

screen_w, screen_h = get_screen_resolution()

# === Redimensionare cu padding ===
def resize_with_padding(img, screen_w, screen_h, padding_percent=0.05):
    max_w = int(screen_w * (1 - 2 * padding_percent))
    max_h = int(screen_h * (1 - 2 * padding_percent))

    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Canvas alb
    canvas = 255 * np.ones((screen_h, screen_w, 3), dtype=np.uint8)
    y_offset = (screen_h - new_h) // 2
    x_offset = (screen_w - new_w) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

# === Funcție casetă rezultate sub imagine ===
def adauga_caseta_rezultate(img, detectii, font_scale=0.7):
    padding = 10
    linii_text = [f"{nume} ({conf:.2f})" for nume, conf in detectii]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    grosime = 2

    # Determină dimensiuni text
    linii_dim = [cv2.getTextSize(text, font, font_scale, grosime)[0] for text in linii_text]
    latime_max = max([dim[0] for dim in linii_dim], default=0)
    inaltime_totala = sum([dim[1] + padding for dim in linii_dim])

    h, w = img.shape[:2]
    box_img = 255 * np.ones((inaltime_totala + 2*padding, w, 3), dtype=np.uint8)

    y_cursor = padding + 5
    for text in linii_text:
        dim = cv2.getTextSize(text, font, font_scale, grosime)[0]
        x = (w - dim[0]) // 2
        cv2.putText(box_img, text, (x, y_cursor + dim[1]), font, font_scale, (0, 0, 0), grosime)
        y_cursor += dim[1] + padding

    # Returnează imaginea cu caseta adăugată dedesubt
    return np.vstack((img, box_img))

# === Procesare imagini ===
def proceseaza_imagini_aleator():
    imagini = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(imagini)

    for filename in imagini:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)

        results = model.predict(source=img, save=False, conf=0.5)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Extragem detectiile pentru caseta de text
        detectii = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                nume = model.names[cls_id]
                detectii.append((nume, conf))

        # Adaugăm caseta de rezultate dedesubt
        img_cu_text = adauga_caseta_rezultate(img, detectii)

        # Redimensionare + padding + centrare
        img_display = resize_with_padding(img_cu_text, screen_w, screen_h)

        win_name = "Rezultat aleator"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(win_name, 0, 0)
        cv2.imshow(win_name, img_display)

        key = cv2.waitKey(0)
        if key == 27:  # ESC
            print("Ieșire...")
            cv2.destroyAllWindows()
            return

        save_path = os.path.join(output_folder, f"rezultat_{filename}")
        cv2.imwrite(save_path, img)

    cv2.destroyAllWindows()

# === Meniu principal ===
while True:
    print("\n===== MENIU =====")
    print("1. Procesează imagini aleatoriu")
    print("2. Ieși din program")
    optiune = input("Alege opțiunea (1/2): ")

    if optiune == "1":
        proceseaza_imagini_aleator()
    elif optiune == "2":
        print("Program încheiat.")
        sys.exit()
    else:
        print("Opțiune invalidă. Încearcă din nou.")
