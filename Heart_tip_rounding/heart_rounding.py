import cv2
import numpy as np
from scipy.optimize import leastsq
import os
import csv
import matplotlib.pyplot as plt

# Ścieżka do folderu z obrazami i wynikami
folder_path = '...'
output_csv_path = '...'
output_images_path = '...'

# Tworzenie folderu na zapisane obrazy, jeśli nie istnieje
os.makedirs(output_images_path, exist_ok=True)

def fit_circle(params, x, y):
    xc, yc, r = params
    return np.sqrt((x - xc)**2 + (y - yc)**2) - r

results = []
region_radius = 15  # ZOSTAŁO WYZNACZONE EMPIRYCZNIE - MOZNA MODYFIKOWAC
stop_program = False

def on_key(event):
    global stop_program
    if event.key == 'q':
        stop_program = True
        plt.close()
    elif event.key == 'enter':
        plt.close()

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.png'):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        cnt_points = cnt[:, 0, :]

        bottom_right = max(cnt_points, key=lambda p: (p[0] + p[1]))
        distances = np.linalg.norm(cnt_points - bottom_right, axis=1)
        region_mask = distances < region_radius
        region_points = cnt_points[region_mask]

        if len(region_points) < 3:
            continue

        x = region_points[:, 0]
        y = region_points[:, 1]
        x_m, y_m = np.mean(x), np.mean(y)
        initial_guess = (x_m, y_m, 10)

        try:
            result, _ = leastsq(fit_circle, initial_guess, args=(x, y))
            xc, yc, radius = result
            results.append([filename, round(radius, 2)])

            output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(output, (int(xc), int(yc)), int(radius), (0, 255, 0), 2)
            cv2.circle(output, tuple(bottom_right), 3, (0, 0, 255), -1)
            for pt in region_points:
                cv2.circle(output, tuple(pt), 1, (255, 0, 0), -1)

            # Zapis obrazu do pliku
            output_img_path = os.path.join(output_images_path, f"out_{filename}")
            cv2.imwrite(output_img_path, output)

            # Wyświetlenie obrazu
            fig = plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.title(f'{filename} - promień: {radius:.2f} px\nEnter = dalej, q = zakończ')
            plt.axis('off')
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()

            if stop_program:
                print("Zamknięcie programu przez użytkownika.")
                break

        except Exception as e:
            print(f"Błąd w pliku {filename}: {e}")

# Zapis CSV z wynikami
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Plik', 'Promień'])
    writer.writerows(results)

print(f"\nWyniki zapisane do {output_csv_path}")
print(f"Obrazy zapisane do folderu: {output_images_path}")
