### 📐 Heart Area vs. Bounding Box Ratio

This metric provides a compact morphological descriptor that relates the size and shape of the heart region to its minimal bounding rectangle. The analysis proceeds as follows:

- The heart region is extracted from the segmentation mask, where heart pixels are defined by a grayscale value of `255`.
- The **heart area** is calculated as the number of pixels belonging to the heart.
- Using OpenCV’s `cv2.boundingRect()`, we compute the **axis-aligned bounding box** that tightly encloses the heart region.
- The area of this rectangle (`width × height`) is then used to compute the **area ratio**:

  \[
  \text{area ratio} = \frac{\text{heart area}}{\text{bounding box area}}
  \]

This ratio offers a measure of how tightly the heart shape fills its bounding box. Lower ratios may indicate elongated or irregular shapes, whereas higher ratios imply a more compact and symmetric form.

Visualizations include the bounding box drawn over the heart mask along with the computed ratio.
