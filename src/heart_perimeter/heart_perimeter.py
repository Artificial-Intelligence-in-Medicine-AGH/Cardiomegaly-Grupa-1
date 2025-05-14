import nibabel as nib
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def calculate_heart_perimeter(file_path,heart_label):
    img = nib.load(file_path)
    data = img.get_fdata()
    
    if data.ndim == 3:
        data = data[:,:,0]

    heart_mask = (data == heart_label).astype(np.uint8)

    contours,_ = cv2.findContours(heart_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(cnt,True) for cnt in contours)
    
    contour_img = cv2.cvtColor((heart_mask * 255), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img,contours,-1,(0,255,0),1)

    plt.figure(figsize=(8,8))
    plt.title("Heart Perimeter")
    plt.imshow(contour_img)
    plt.axis("off")
    plt.show()

    return perimeter

def read_data(folder_path,heart_label):
    results = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(folder_path,filename)
            perimeter = calculate_heart_perimeter(file_path,heart_label)
            results[filename] = perimeter
            print(f"{filename}: Heart Perimeter = {perimeter}")

    return results

folder_path = "DATA"
heart_label = 2
perimeter_results = read_data(folder_path,heart_label)