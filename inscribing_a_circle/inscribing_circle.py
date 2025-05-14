import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def analyze_heart_center(mask):
    heart_mask = (mask == 255).astype(np.uint8)
    lung_mask = (mask == 128).astype(np.uint8)

    dist_transform = distance_transform_edt(heart_mask)
    max_dist = np.max(dist_transform)
    center_coords = np.unravel_index(np.argmax(dist_transform), dist_transform.shape)
    center_x = center_coords[1]

    # Dijkstra przez środek
    start_points = np.argwhere(heart_mask[0, :] > 0)
    end_points = np.argwhere(heart_mask[-1, :] > 0)
    min_path = None
    min_cost = np.inf

    for sp in start_points:
        for ep in end_points:
            start = (0, sp[0])
            end = (heart_mask.shape[0] - 1, ep[0])
            try:
                path, cost = route_through_array(-dist_transform, start, end, fully_connected=True)
                if cost < min_cost:
                    min_cost = cost
                    min_path = path
            except:
                continue

    path = np.array(min_path) if min_path is not None else np.array([])

    # Szerokość serca
    heart_columns = np.any(heart_mask, axis=0)
    heart_indices = np.where(heart_columns)[0]
    heart_width = float(np.sum(heart_columns))
    heart_left = heart_indices.min() if heart_indices.size > 0 else 0
    heart_right = heart_indices.max() if heart_indices.size > 0 else 0

    # Szerokość płuc
    lung_mask_rgb = cv2.inRange(mask, 127, 129)
    lung_points = np.argwhere(lung_mask_rgb > 0)
    if lung_points.size > 0:
        x_coords = lung_points[:, 1]
        lungs_width = float(x_coords.max() - x_coords.min())
        diaphragm_y = int(np.mean(lung_points[:, 0]))
    else:
        lungs_width = 0.0
        diaphragm_y = mask.shape[0] - 1

    # Powierzchnia serca po lewej/prawej stronie od środka
    left_area = np.sum(heart_mask[:, :center_x])
    right_area = np.sum(heart_mask[:, center_x:])
    side_ratio = left_area / right_area if right_area > 0 else np.nan

    # Piksele w dolnych rogach serca (poniżej środka i poniżej promienia)
    # lower_heart = heart_mask[center_coords[0]+int(max_dist):, :]
    # left_lower_pixels = np.sum(lower_heart[:, :center_x])
    # right_lower_pixels = np.sum(lower_heart[:, center_x:])
    # lower_ratio = left_lower_pixels / right_lower_pixels if right_lower_pixels > 0 else np.nan

    # Obliczenia dla okręgu wpisanego
    inscribed_radius = max_dist
    inscribed_area = np.pi * (inscribed_radius ** 2)

    return {
        "center_y": center_coords[0],
        "center_x": center_x,
        "radius": max_dist,
        "path": path,
        "heart_width": heart_width,
        "lungs_width": lungs_width,
        "diaphragm_y": diaphragm_y,
        "left_area": left_area,
        "right_area": right_area,
        "side_ratio": side_ratio,
        "left_lower_pixels": left_lower_pixels,
        "right_lower_pixels": right_lower_pixels,
        "lower_ratio": lower_ratio,
        "heart_left": heart_left,
        "heart_right": heart_right,
        "inscribed_radius": inscribed_radius,
        "inscribed_area": inscribed_area
    }

def batch_process(folder_path, output_csv="wyniki.csv", save_plots=True):
    results = []

    if save_plots and not os.path.exists("wizualizacje"):
        os.makedirs("wizualizacje")

    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder_path, filename)
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            data = analyze_heart_center(mask)
            results.append({
                "filename": filename,
                "center_x": data["center_x"],
                "center_y": data["center_y"],
                "radius": data["radius"],
                "heart_width": data["heart_width"],
                "lungs_width": data["lungs_width"],
                "left_area": data["left_area"],
                "right_area": data["right_area"],
                "side_ratio": data["side_ratio"],
                "left_lower_pixels": data["left_lower_pixels"],
                "right_lower_pixels": data["right_lower_pixels"],
                "lower_ratio": data["lower_ratio"],
                "inscribed_radius": data["inscribed_radius"],
                "inscribed_area": data["inscribed_area"]
            })

            if save_plots:
                plt.figure(figsize=(4, 4))
                plt.imshow(mask, cmap='gray')

                if data["path"].size > 0:
                    plt.plot(data["path"][:, 1], data["path"][:, 0], 'orange', label="Dijkstra")

                plt.plot(data["center_x"], data["center_y"], 'ro', label="Środek")
                plt.axvline(x=data["center_x"], color='orange', linestyle='-', linewidth=2, label="Linia pionowa")
                plt.gca().add_patch(plt.Circle(
                    (data["center_x"], data["center_y"]),
                    data["radius"], color='r', fill=False, linestyle='--', label="Koło"
                ))

                lung_mask_rgb = cv2.inRange(mask, 127, 129)
                lung_points = np.argwhere(lung_mask_rgb > 0)
                if lung_points.size > 0:
                    x_coords = lung_points[:, 1]
                    y_mean = int(np.mean(lung_points[:, 0]))
                    x1, x2 = x_coords.min(), x_coords.max()
                    plt.plot([x1, x2], [y_mean, y_mean], color='red', linewidth=2, label="Szerokość płuc")

                # Rysowanie linii szerokości serca
                heart_y = int(data["center_y"] + data["radius"] / 2)
                plt.plot([data["heart_left"], data["heart_right"]], [heart_y, heart_y], color='magenta', linewidth=2, label="Szerokość serca")

                plt.axis('off')
                plt.legend()
                plt.title(filename)
                plt.savefig(os.path.join("wizualizacje", f"{filename}_vis.png"))
                plt.close()

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, float_format='%.5f')
    print(f"Zapisano dane do {output_csv}")

if __name__ == "__main__":
    batch_process(folder_path="maski", output_csv="wyniki.csv", save_plots=True)
