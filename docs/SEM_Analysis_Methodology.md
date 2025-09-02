# Methodology for Quantitative SEM Surface Morphology Analysis

## Overview

This methodology describes a comprehensive quantitative approach for analyzing surface morphological changes in soil particles using Scanning Electron Microscopy (SEM) images. The method employs advanced digital image processing techniques to extract objective, measurable parameters from SEM micrographs.

## Equipment and Software Requirements

### Hardware
- Scanning Electron Microscope (SEM) with digital imaging capability
- Computer with minimum 8GB RAM for image processing

### Software Dependencies
- Python 3.8 or higher
- Required Python packages:
  - opencv-python (cv2)
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-image (skimage)
  - scipy
  - pathlib

## Sample Preparation

1. **Sample mounting**: Standard SEM sample preparation protocols
2. **Imaging conditions**: 
   - Optimal magnification: 2μm scale (provides balance between resolution and field of view)
   - Accelerating voltage: 15-20 kV (typical for soil samples)
   - Working distance: 8-12 mm
   - Image format: TIFF (uncompressed, 8-bit grayscale)

## Image Acquisition Protocol

1. **Systematic sampling**: Capture multiple representative areas per sample
2. **Standardized conditions**: Maintain consistent brightness, contrast, and focus
3. **Scale bar inclusion**: Ensure scale bar is visible and unobstructed
4. **Image naming**: Use systematic naming convention (e.g., SoilType_Treatment_ImageNumber)

## Analytical Methodology

### 1. Surface Roughness Analysis

**Principle**: Quantifies surface texture variations using statistical measures of pixel intensity distribution.

**Parameters measured**:
- Surface variation (standard deviation of pixel intensities)
- Texture contrast (GLCM-based contrast measure)
- Texture homogeneity
- Texture energy
- Local variance

**Method**:
```python
# Calculate GLCM properties for texture analysis
from skimage.feature import graycomatrix, graycoprops

glcm = graycomatrix(image, distances=[1,2,3], angles=[0,45,90,135])
contrast = graycoprops(glcm, 'contrast').mean()
homogeneity = graycoprops(glcm, 'homogeneity').mean()
```

### 2. Particle Aggregation Analysis

**Principle**: Analyzes particle size distribution and clustering patterns to assess aggregation state.

**Parameters measured**:
- Aggregation index (mean particle area)
- Size heterogeneity (coefficient of variation)
- Particle density (particles per unit area)
- Mean compactness

**Method**:
```python
# Automated particle detection using adaptive thresholding
binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv2.THRESH_BINARY_INV, 11, 2)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### 3. Porosity and Void Space Analysis

**Principle**: Identifies and quantifies pore spaces and void areas within the sample.

**Parameters measured**:
- Total porosity (fraction of void space)
- Pore count and size distribution
- Pore density

**Method**:
```python
# Pore detection by image inversion and thresholding
img_inv = 255 - image
_, pore_binary = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

### 4. Boundary Definition Analysis

**Principle**: Assesses particle boundary clarity and definition using edge detection.

**Parameters measured**:
- Boundary sharpness
- Boundary continuity
- Edge density
- Gradient magnitude

**Method**:
```python
# Multi-method edge detection
edges_canny = cv2.Canny(image, 50, 150)
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
```

## Statistical Analysis

### Data Processing
1. **Normalization**: All measurements converted to standard units (μm, μm²)
2. **Quality control**: Automated filtering of artifacts and measurement errors
3. **Statistical validation**: Calculation of means, standard deviations, and confidence intervals

### Comparative Analysis
1. **Non-parametric testing**: Mann-Whitney U tests for between-group comparisons
2. **Effect size calculation**: Percentage changes and Cohen's d values
3. **Multiple comparisons**: Bonferroni correction when applicable

## Quality Assurance

### Image Quality Criteria
- Minimum resolution: 1024×1024 pixels
- Adequate contrast and brightness
- Minimal noise and artifacts
- Clear scale bar visibility

### Measurement Validation
- Manual verification of automated measurements (subsample)
- Inter-operator reproducibility testing
- Cross-validation with alternative methods

## Data Reporting

### Standard Metrics
- Sample size (number of images and particles analyzed)
- Mean ± standard deviation for each parameter
- Statistical significance values (p-values)
- Effect sizes and percentage changes

### Visualization
- Box plots for group comparisons
- Histograms for distribution analysis
- Scatter plots for correlation analysis
- Publication-quality figures (300 DPI minimum)

## Limitations and Considerations

1. **Scale dependency**: Results may vary with magnification level
2. **Sample preparation effects**: Coating thickness and uniformity
3. **Imaging conditions**: Consistency critical for comparative studies
4. **Automated vs manual**: Validation of automated measurements recommended

## Applications

This methodology is applicable for:
- Soil treatment effectiveness assessment
- Surface modification studies
- Environmental remediation monitoring
- Material science surface characterization
- Geological sample analysis

## Citation and Reproducibility

When using this methodology, please cite:
- This methodology document
- Software packages used (OpenCV, scikit-image, etc.)
- Specific parameter settings and modifications

## References

1. Haralick, R.M., et al. (1973). Textural features for image classification. IEEE Transactions on Systems, Man, and Cybernetics.
2. Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics.
3. Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence.

---

**Corresponding Author**: Hafiz Asad Ullah Sajid  
**Institution**: Sichuan Agricultural University, Department of Environmental Engineering  
**Email**: hafizasadullahsajid.iub@gmail.com  
**Date**: September 2025  
**Version**: 1.0
