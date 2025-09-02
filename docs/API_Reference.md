# API Reference

**Author**: Hafiz Asad Ullah Sajid  
**Institution**: Sichuan Agricultural University, Department of Environmental Engineering  
**Email**: hafizasadullahsajid.iub@gmail.com  
**GitHub**: https://github.com/asadullahsajid/SEM-Soil-Morphology-Analyzer

## Classes

### SEMMorphologyAnalyzer

Main class for SEM surface morphology analysis.

#### `__init__(self, config: Optional[Dict] = None)`

Initialize the analyzer with optional configuration.

**Parameters:**
- `config` (dict, optional): Configuration dictionary. If None, uses default configuration.

**Example:**
```python
analyzer = SEMMorphologyAnalyzer()

# Or with custom config
config = {'scale_pixels_per_um': 150}
analyzer = SEMMorphologyAnalyzer(config)
```

#### `analyze_surface_roughness(self, image_path: Path) -> Dict`

Analyze surface roughness and texture properties.

**Parameters:**
- `image_path` (Path): Path to the SEM image file

**Returns:**
- `dict`: Dictionary containing roughness metrics including:
  - `texture_contrast`: GLCM-based contrast measure
  - `texture_homogeneity`: Texture uniformity
  - `surface_variation`: Pixel intensity standard deviation
  - `edge_density`: Boundary definition measure

#### `analyze_particle_aggregation(self, image_path: Path) -> Dict`

Analyze particle aggregation and size distribution.

**Parameters:**
- `image_path` (Path): Path to the SEM image file

**Returns:**
- `dict`: Dictionary containing aggregation metrics including:
  - `particle_count`: Number of detected particles
  - `aggregation_index`: Mean particle area
  - `size_heterogeneity`: Coefficient of variation of particle sizes
  - `particle_density`: Particles per unit area

#### `analyze_porosity(self, image_path: Path) -> Dict`

Analyze pore structure and void spaces.

**Parameters:**
- `image_path` (Path): Path to the SEM image file

**Returns:**
- `dict`: Dictionary containing porosity metrics including:
  - `porosity`: Fraction of void space
  - `pore_count`: Number of detected pores
  - `mean_pore_size`: Average pore area

#### `analyze_boundary_definition(self, image_path: Path) -> Dict`

Analyze particle boundary definition and clarity.

**Parameters:**
- `image_path` (Path): Path to the SEM image file

**Returns:**
- `dict`: Dictionary containing boundary metrics including:
  - `boundary_sharpness`: Edge detection variance
  - `boundary_continuity`: Edge pixel density

#### `analyze_image(self, image_path: Path, condition: str = "", metadata: Dict = None) -> Dict`

Perform comprehensive analysis on a single SEM image.

**Parameters:**
- `image_path` (Path): Path to the SEM image file
- `condition` (str, optional): Experimental condition label
- `metadata` (dict, optional): Additional metadata

**Returns:**
- `dict`: Complete analysis results combining all metrics

#### `analyze_dataset(self, input_dir: Path, condition_mapping: Dict = None) -> pd.DataFrame`

Analyze a complete dataset of SEM images.

**Parameters:**
- `input_dir` (Path): Directory containing SEM images
- `condition_mapping` (dict, optional): Mapping of subfolder names to conditions

**Returns:**
- `pd.DataFrame`: DataFrame containing all analysis results

#### `create_visualizations(self, df: pd.DataFrame, output_dir: Path)`

Create comprehensive visualizations of analysis results.

**Parameters:**
- `df` (pd.DataFrame): Analysis results DataFrame
- `output_dir` (Path): Directory to save visualizations

#### `generate_report(self, df: pd.DataFrame, output_dir: Path)`

Generate a comprehensive analysis report.

**Parameters:**
- `df` (pd.DataFrame): Analysis results DataFrame
- `output_dir` (Path): Directory to save the report

## Configuration Parameters

### Default Configuration

```python
{
    'scale_pixels_per_um': 100,        # Scale calibration
    'min_particle_area_um2': 0.01,     # Minimum particle size
    'max_particle_area_um2': 100,      # Maximum particle size
    'gaussian_blur_kernel': (3, 3),    # Preprocessing blur
    'morphology_kernel_size': 3,       # Morphological operations
    'adaptive_threshold_block_size': 11, # Thresholding
    'adaptive_threshold_c': 2,         # Thresholding constant
    'canny_threshold1': 50,            # Edge detection
    'canny_threshold2': 150,           # Edge detection
    'glcm_distances': [1, 2, 3],       # GLCM distances
    'glcm_angles': [0, 45, 90, 135]    # GLCM angles
}
```

### Customization Examples

```python
# High-resolution analysis
high_res_config = {
    'scale_pixels_per_um': 200,
    'min_particle_area_um2': 0.005,
    'canny_threshold1': 30,
    'canny_threshold2': 100
}

# Noise-tolerant analysis
noise_tolerant_config = {
    'gaussian_blur_kernel': (5, 5),
    'morphology_kernel_size': 5,
    'adaptive_threshold_block_size': 15
}
```

## Command Line Interface

### Basic Usage

```bash
python sem_morphology_analyzer.py --input_dir /path/to/images --output_dir /path/to/results
```

### With Configuration

```bash
python sem_morphology_analyzer.py \
  --input_dir /path/to/images \
  --output_dir /path/to/results \
  --config config.json \
  --condition_mapping conditions.json
```

### Arguments

- `--input_dir`: Directory containing SEM images (required)
- `--output_dir`: Directory to save results (required)
- `--config`: Path to JSON configuration file (optional)
- `--condition_mapping`: Path to JSON condition mapping file (optional)
