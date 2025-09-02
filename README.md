# SEM Surface Morphology Analyzer

A comprehensive Python toolkit for quantitative analysis of Scanning Electron Microscopy (SEM) images to extract surface morphology parameters for soil science, materials science, and environmental research.

## Features

- **Automated Surface Roughness Analysis**: GLCM-based texture analysis, surface variation quantification
- **Particle Aggregation Assessment**: Size distribution, heterogeneity, density measurements
- **Porosity and Void Space Analysis**: Automated pore detection and quantification
- **Boundary Definition Analysis**: Edge detection and particle boundary clarity assessment
- **Statistical Analysis**: Comparative statistics and significance testing
- **Publication-Quality Visualizations**: Automated generation of research-grade figures

## Installation

### Requirements
- Python 3.8 or higher
- Required packages (install via pip):

```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-image scipy
```

### Quick Start

1. Clone or download this repository:
```bash
git clone https://github.com/asadullahsajid/SEM-Soil-Morphology-Analyzer.git
cd SEM-Soil-Morphology-Analyzer
```

2. Install required packages
3. Run the analyzer on your SEM images:

```bash
python src/sem_morphology_analyzer.py --input_dir /path/to/images --output_dir /path/to/results
```

## Usage

### Basic Usage

```python
from sem_morphology_analyzer import SEMMorphologyAnalyzer
from pathlib import Path

# Initialize analyzer
analyzer = SEMMorphologyAnalyzer()

# Analyze a single image
result = analyzer.analyze_image(Path("sample_image.tif"), condition="treatment_1")

# Analyze a complete dataset
df = analyzer.analyze_dataset(Path("image_directory"))

# Create visualizations
analyzer.create_visualizations(df, Path("output_directory"))

# Generate report
analyzer.generate_report(df, Path("output_directory"))
```

### Advanced Usage with Configuration

```python
# Custom configuration
config = {
    'scale_pixels_per_um': 150,  # Adjust for your magnification
    'min_particle_area_um2': 0.005,
    'max_particle_area_um2': 200,
    'canny_threshold1': 30,
    'canny_threshold2': 100
}

analyzer = SEMMorphologyAnalyzer(config)
```

### Command Line Usage

```bash
# Basic analysis
python sem_morphology_analyzer.py --input_dir ./images --output_dir ./results

# With custom configuration
python sem_morphology_analyzer.py --input_dir ./images --output_dir ./results --config config.json

# With condition mapping for experimental design
python sem_morphology_analyzer.py --input_dir ./images --output_dir ./results --condition_mapping conditions.json
```

## Configuration

### Default Parameters

```json
{
    "scale_pixels_per_um": 100,
    "min_particle_area_um2": 0.01,
    "max_particle_area_um2": 100,
    "gaussian_blur_kernel": [3, 3],
    "morphology_kernel_size": 3,
    "adaptive_threshold_block_size": 11,
    "adaptive_threshold_c": 2,
    "canny_threshold1": 50,
    "canny_threshold2": 150,
    "glcm_distances": [1, 2, 3],
    "glcm_angles": [0, 45, 90, 135]
}
```

### Condition Mapping Example

```json
{
    "before_treatment": "Control",
    "after_treatment_1": "Treatment A",
    "after_treatment_2": "Treatment B"
}
```

## Output Parameters

### Surface Roughness
- `texture_contrast`: GLCM-based contrast measure
- `texture_homogeneity`: Texture uniformity
- `surface_variation`: Pixel intensity standard deviation
- `edge_density`: Boundary definition measure

### Particle Aggregation
- `aggregation_index`: Mean particle area
- `size_heterogeneity`: Coefficient of variation of particle sizes
- `particle_density`: Particles per unit area
- `mean_compactness`: Average particle shape compactness

### Porosity
- `porosity`: Fraction of void space
- `pore_count`: Number of detected pores
- `mean_pore_size`: Average pore area
- `pore_density`: Pores per unit area

### Boundary Definition
- `boundary_sharpness`: Edge detection variance
- `boundary_continuity`: Edge pixel density
- `mean_gradient`: Average gradient magnitude
- `edge_strength`: Sobel edge strength

## Image Requirements

- **Format**: TIFF preferred (uncompressed, 8-bit grayscale)
- **Resolution**: Minimum 1024×1024 pixels
- **Quality**: Clear focus, adequate contrast
- **Scale**: Consistent magnification within dataset
- **Naming**: Systematic file naming recommended

## Best Practices

1. **Consistent Imaging**: Use identical SEM settings for comparative studies
2. **Scale Calibration**: Verify and adjust `scale_pixels_per_um` parameter
3. **Quality Control**: Review automated measurements for accuracy
4. **Statistical Power**: Analyze sufficient number of images per condition (≥5 recommended)
5. **Documentation**: Record all analysis parameters and settings

## Validation

This toolkit has been validated against:
- Manual particle measurements
- Alternative image analysis software
- Published morphological studies

## Applications

- Soil treatment effectiveness assessment
- Surface modification studies
- Environmental remediation monitoring
- Material characterization
- Geological sample analysis
- Nanoparticle analysis

## Citation

If you use this tool in your research, please cite:

```
Sajid, H. A. U. (2025). SEM Surface Morphology Analyzer: A Python toolkit for quantitative 
surface analysis of electron microscopy images for soil treatment effectiveness assessment. 
GitHub repository. https://github.com/asadullahsajid/SEM-Soil-Morphology-Analyzer
```

## License

MIT License - see LICENSE file for details

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Support

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Email: hafizasadullahsajid.iub@gmail.com

## Acknowledgments

- OpenCV community for image processing algorithms
- scikit-image developers for advanced image analysis tools
- Scientific Python ecosystem (NumPy, SciPy, pandas, matplotlib)

## Version History

- **v1.0** (September 2025): Initial release with comprehensive morphology analysis

---

**Authors**: Hafiz Asad Ullah Sajid  
**Institution**: Sichuan Agricultural University, Department of Environmental Engineering  
**Contact**: hafizasadullahsajid.iub@gmail.com
