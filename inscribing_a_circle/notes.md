### 🫀 Inscribed Circle Algorithm – Heart Center Analysis

As part of our heart morphology analysis, we implemented an **inscribed circle detection** algorithm to identify the geometric center and estimate the internal structure of the heart mask. Here's how it works:

- The binary heart region is extracted from the segmentation mask.
- Using a **Euclidean Distance Transform**, we compute the distance from every pixel within the heart mask to the nearest background pixel.
- The **maximum value** of this distance map indicates the **radius of the largest circle** that can be inscribed entirely within the heart region.
- The coordinates of this maximum point define the **center of the heart**.
- This circle provides a stable reference for assessing symmetry, estimating area, and generating features such as left/right area ratio, vertical alignment, and anatomical proportions.

This method is robust against noise and non-uniform shapes, offering a meaningful approximation of the heart's internal structure for further quantitative analysis.
