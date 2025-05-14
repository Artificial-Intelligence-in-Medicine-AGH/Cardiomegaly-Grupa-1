import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate_heart_to_box_ratio(mask):
    # Maska serca (przyjmujemy że serce to piksele o wartości 255)
    heart_mask = (mask == 255).astype(np.uint8)

    # Oblicz pole serca
    heart_area = int(np.sum(heart_mask))

    # Wyznacz prostokąt opisany (bounding box)
    x, y, w, h = cv2.boundingRect(heart_mask)

    # Oblicz pole prostokąta
    bounding_box_area = int(w * h)

    # Stosunek pola serca do prostokąta
    area_ratio = heart_area / bounding_box_area if bounding_box_area > 0 else np.nan

    return {
        "heart_area": heart_area,
        "bounding_box_area": bounding_box_area,
        "area_ratio": area_ratio,
        "bbox": (x, y, w, h)
    }

def batch_process_heart_ratios(folder_path, output_csv="heart_box_ratios.csv", save_visuals=True):
    results = []

    vis_dir = "heart_box_visuals"
    if save_visuals and not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(folder_path, filename)
            mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Błąd przy wczytywaniu: {filename}")
                continue

            data = calculate_heart_to_box_ratio(mask)

            results.append({
                "filename": filename,
                "heart_area": data["heart_area"],
                "bounding_box_area": data["bounding_box_area"],
                "area_ratio": data["area_ratio"]
            })

            if save_visuals:
                # Tworzenie wizualizacji
                plt.figure(figsize=(4, 4))
                plt.imshow(mask, cmap="gray")
                x, y, w, h = data["bbox"]
                plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor='r', facecolor='none', linewidth=2))
                plt.title(f"{filename}\nStosunek: {data['area_ratio']:.3f}")
                plt.axis('off')
                plt.savefig(os.path.join(vis_dir, f"{filename}_bbox.png"))
                plt.close()

    # Zapis do CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Zapisano wyniki do pliku: {output_csv}")

if __name__ == "__main__":
    # Przykładowe wywołanie
    batch_process_heart_ratios(folder_path="maski", output_csv="heart_box_ratios.csv", save_visuals=True)
