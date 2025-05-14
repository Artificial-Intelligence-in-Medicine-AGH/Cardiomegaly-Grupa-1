# Heart Apex Circle Fitting Algorithm

## Algorithm 

1. **Detect the heart contour**  
   The algorithm begins by identifying the outline of the heart from the medical image.

2. **Find the lower-left apex of the heart**  
   The apex (tip) of the heart, typically located in the lower-left region, is localized.

3. **Select a region near the apex**  
   A specific area around the apex is selected for further analysis.

4. **Fit a circle using the least squares method**  
   A circle is fitted to the contour within the selected region using the **least squares method**:
   - `fit_circle` computes the deviation of contour points from an ideal circle.
   - `leastsq` finds circle parameters that minimize the total error.

## 📊 Results Interpretation

- **Small radius** → Higher likelihood of a **healthy heart**
- **Large radius** → May indicate a **dilated heart** or **bulging apex**, suggesting a higher risk of **cardiomegaly**
