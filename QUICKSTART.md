# Quick Start Guide

## Installation

### Option 1: Direct Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/[your-username]/sem-morphology-analyzer.git
cd sem-morphology-analyzer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Option 2: Manual Installation

1. Download the repository as a ZIP file
2. Extract to your desired location
3. Install dependencies:
```bash
pip install opencv-python numpy pandas matplotlib seaborn scikit-image scipy
```

## Quick Start

### 1. Basic Analysis

```python
from src.sem_morphology_analyzer import SEMMorphologyAnalyzer
from pathlib import Path

# Initialize analyzer
analyzer = SEMMorphologyAnalyzer()

# Analyze a single image
result = analyzer.analyze_image(Path("your_image.tif"), condition="sample_1")
print(f"Surface variation: {result['surface_variation']:.3f}")
print(f"Particle count: {result['particle_count']}")
```

### 2. Command Line Usage

```bash
# Basic analysis
python src/sem_morphology_analyzer.py --input_dir ./images --output_dir ./results

# With custom configuration
python src/sem_morphology_analyzer.py \
  --input_dir ./images \
  --output_dir ./results \
  --config examples/config_example.json
```

### 3. Dataset Analysis

```python
# Analyze multiple conditions
condition_mapping = {
    "before": "Control",
    "after_treatment": "Treated"
}

df = analyzer.analyze_dataset(Path("image_directory"), condition_mapping)

# Generate visualizations and report
analyzer.create_visualizations(df, Path("output"))
analyzer.generate_report(df, Path("output"))
```

## Configuration

Copy and modify `examples/config_example.json`:

```json
{
    "scale_pixels_per_um": 100,
    "min_particle_area_um2": 0.01,
    "max_particle_area_um2": 100
}
```

## Image Requirements

- **Format**: TIFF preferred (8-bit grayscale)
- **Size**: Minimum 512x512 pixels
- **Quality**: Clear focus, good contrast
- **Scale**: Consistent magnification

## Output

The analyzer generates:
- CSV files with quantitative measurements
- Statistical summaries
- Publication-quality plots
- Comprehensive analysis reports

## Need Help?

- Check the [API Reference](docs/API_Reference.md)
- Read the [Methodology](docs/SEM_Analysis_Methodology.md)
- Run the [example script](examples/example_usage.py)
- Open an issue on GitHub

## Citation

```bibtex
@software{sem_morphology_analyzer,
  title={SEM Surface Morphology Analyzer},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-username]/sem-morphology-analyzer}
}
```
