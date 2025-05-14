# Cardiomegaly-Grupa-1
Wyniki pracy grupy 1 w projekcie kardiomegalii


Jasne! Poniżej znajduje się całość Twojego opisu w sformatowanej wersji Markdown — gotowa do wklejenia do pliku `README.md` w repozytorium GitHub:

---

# 🫀 Cardiomegaly Detection from Chest X-rays Using Machine Learning

## 📌 Overview

**Cardiomegaly**, or an enlarged heart, is a radiological sign that may indicate various cardiovascular diseases. Detecting cardiomegaly early from chest X-rays can help with timely diagnosis and intervention.
This project, developed within the **AI MED scientific club**, focuses on building a machine learning pipeline to detect cardiomegaly from chest radiographs (X-rays).

---

## 🧠 Project Summary

The goal of this project is to:

* Segment lungs and heart regions from X-ray images
* Extract geometrical features from segmented masks
* Train classical machine learning models to classify the presence of cardiomegaly
* Evaluate the model's performance and save results for analysis

The pipeline was fully developed in **Python**, leveraging libraries such as `scikit-learn`, `OpenCV`, `NumPy`, and `Pandas`.

---

## 🧪 Dataset

* **Total samples**: 39 annotated chest X-ray images

  * 10 healthy
  * 29 diseased

Each image was processed in two steps:

1. **Segmentation** using deep learning-based models to extract lungs and heart regions
2. **Feature extraction** (lung width, heart width, cardiothoracic ratio, and more) for classification purposes

---

## 🧰 Technologies Used

* Python 3
* `scikit-learn` for classification (Random Forest, K-Nearest Neighbors, SVM, Soft Voting Classifier)
* `OpenCV` for image processing
* `Matplotlib` + `Seaborn` for visualization
* `Pandas` + `ExcelWriter` for logging results
* Jupyter Notebook for prototyping and analysis

---

## 🧮 Machine Learning Approach

### 📑 Feature Extraction

From each segmented chest X-ray image, we calculated a variety of geometrical features describing the lungs and heart. Key features include:

* **Lung width**
* **Heart width**
* **Cardiothoracic ratio (CTR)** — calculated as `heart_width / lung_width`
* **Heart tip rounding** — curvature measurement of the bottom heart edge
* **Heart area to bounding box ratio** — ratio of the segmented heart area to its enclosing rectangle
* **Heart perimeter** — contour length of the segmented heart region

📄 *Additional extracted features are listed in the accompanying Excel file: `calculated_features.xlsx`, included in the repository.*

These features were used as inputs for classical machine learning classifiers.

---

### 🧠 Algorithms Used

This project utilized several supervised learning techniques implemented via `scikit-learn`, including both individual classifiers and ensemble methods:

* **Random Forest (RF)**
  An ensemble of decision trees trained on random subsets of the data and features. Improves accuracy and reduces overfitting compared to single trees.

* **Decision Tree (DT)**
  A simple tree-based classifier that splits data based on feature thresholds. Easy to interpret but prone to overfitting on small datasets.

* **Support Vector Classifier (SVC)**
  A powerful model that finds the optimal hyperplane to separate classes. Works well for small and high-dimensional datasets.

* **K-Nearest Neighbors (KNN)**
  A non-parametric method that classifies new samples based on the majority label among their *k* nearest neighbors in the feature space.

* **Voting Classifier (Ensemble Learning)**
  A meta-model that combines multiple base classifiers (RF, SVC, KNN) using **soft voting** — averaging predicted class probabilities to improve robustness and generalization.

---

### 🧪 Validation Strategy

* **Stratified K-Fold Cross-Validation**
  Ensures each fold maintains the original distribution of the target labels (e.g., cardiomegaly vs. normal), which is crucial for small or imbalanced datasets.

* **Standard K-Fold Cross-Validation**
  Splits data into *k* equally sized parts; each fold is used as a test set once. Useful for evaluating model stability across different subsets.

These strategies were used to evaluate generalization performance and to reduce the risk of overfitting given the small dataset size (39 samples).

---

