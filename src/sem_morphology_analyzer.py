"""
SEM Surface Morphology Analyzer - Public Research Tool
=====================================================

A comprehensive Python tool for quantitative analysis of SEM (Scanning Electron Microscopy) images
to extract surface morphology parameters for soil science, materials science, and environmental research.

Authors: [Your Research Team]
Institution: [Your Institution]
Version: 1.0
Date: September 2025
License: MIT License

Requirements:
- Python 3.8+
- opencv-python
- numpy
- pandas
- matplotlib
- seaborn
- scikit-image
- scipy

Usage:
    python sem_morphology_analyzer.py --input_dir /path/to/images --output_dir /path/to/results

Features:
- Automated surface roughness analysis
- Particle aggregation assessment
- Porosity and void space quantification
- Boundary definition analysis
- Statistical comparison between conditions
- Publication-quality visualizations

Citation:
If you use this tool in your research, please cite:
[Your paper citation will go here]
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json
from typing import Dict, List, Tuple, Optional
from skimage.feature import graycomatrix, graycoprops
from skimage import filters, morphology
from scipy import stats
from scipy.stats import mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

class SEMMorphologyAnalyzer:
    """
    Comprehensive SEM surface morphology analyzer for research applications.
    
    This class provides methods for quantitative analysis of SEM images including:
    - Surface roughness and texture analysis
    - Particle aggregation state assessment
    - Porosity and pore structure analysis
    - Particle boundary definition analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the analyzer with configuration parameters.
        
        Args:
            config: Dictionary containing analysis parameters
        """
        self.config = config or self._default_config()
        self.results = []
        
    def _default_config(self) -> Dict:
        """Default configuration parameters"""
        return {
            'scale_pixels_per_um': 100,  # Adjust based on your magnification
            'min_particle_area_um2': 0.01,
            'max_particle_area_um2': 100,
            'gaussian_blur_kernel': (3, 3),
            'morphology_kernel_size': 3,
            'adaptive_threshold_block_size': 11,
            'adaptive_threshold_c': 2,
            'canny_threshold1': 50,
            'canny_threshold2': 150,
            'glcm_distances': [1, 2, 3],
            'glcm_angles': [0, 45, 90, 135]
        }
    
    def analyze_surface_roughness(self, image_path: Path) -> Dict:
        """
        Analyze surface roughness and texture properties.
        
        Args:
            image_path: Path to the SEM image
            
        Returns:
            Dictionary containing roughness and texture metrics
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {}
            
            # Resize for consistent analysis
            img = cv2.resize(img, (512, 512))
            
            # Calculate GLCM properties
            glcm = graycomatrix(
                img, 
                distances=self.config['glcm_distances'],
                angles=self.config['glcm_angles'],
                levels=256, 
                symmetric=True, 
                normed=True
            )
            
            # Extract texture properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            # Surface variation (roughness indicator)
            surface_variation = np.std(img)
            
            # Edge density (boundary sharpness)
            edges = cv2.Canny(img, self.config['canny_threshold1'], self.config['canny_threshold2'])
            edge_density = np.sum(edges > 0) / edges.size
            
            # Local variance calculation
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            img_mean = cv2.filter2D(img.astype(np.float32), -1, kernel)
            img_sq_mean = cv2.filter2D((img.astype(np.float32))**2, -1, kernel)
            local_variance = (img_sq_mean - img_mean**2).mean()
            
            return {
                'texture_contrast': contrast,
                'texture_dissimilarity': dissimilarity,
                'texture_homogeneity': homogeneity,
                'texture_energy': energy,
                'texture_correlation': correlation,
                'surface_variation': surface_variation,
                'edge_density': edge_density,
                'local_variance': local_variance
            }
            
        except Exception as e:
            print(f"Error analyzing surface roughness for {image_path}: {e}")
            return {}
    
    def analyze_particle_aggregation(self, image_path: Path) -> Dict:
        """
        Analyze particle aggregation and size distribution.
        
        Args:
            image_path: Path to the SEM image
            
        Returns:
            Dictionary containing aggregation metrics
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {}
            
            # Preprocessing
            img_blur = cv2.GaussianBlur(img, self.config['gaussian_blur_kernel'], 0)
            img_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img_blur)
            
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(
                img_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 
                self.config['adaptive_threshold_block_size'],
                self.config['adaptive_threshold_c']
            )
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.config['morphology_kernel_size'], 
                                              self.config['morphology_kernel_size']))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find particles
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate particle properties
            areas_um2 = []
            compactness_values = []
            
            for contour in contours:
                area_pixels = cv2.contourArea(contour)
                area_um2 = area_pixels / (self.config['scale_pixels_per_um'] ** 2)
                
                # Filter by size
                if self.config['min_particle_area_um2'] <= area_um2 <= self.config['max_particle_area_um2']:
                    areas_um2.append(area_um2)
                    
                    # Calculate compactness
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        compactness = (4 * np.pi * area_pixels) / (perimeter ** 2)
                        compactness_values.append(min(compactness, 1.0))
            
            if not areas_um2:
                return {}
            
            # Aggregation metrics
            aggregation_index = np.mean(areas_um2)
            size_heterogeneity = np.std(areas_um2) / np.mean(areas_um2) if np.mean(areas_um2) > 0 else 0
            particle_density = len(areas_um2) / ((img.shape[0] * img.shape[1]) / (self.config['scale_pixels_per_um'] ** 2))
            mean_compactness = np.mean(compactness_values) if compactness_values else 0
            
            return {
                'particle_count': len(areas_um2),
                'aggregation_index': aggregation_index,
                'size_heterogeneity': size_heterogeneity,
                'particle_density': particle_density,
                'mean_compactness': mean_compactness,
                'mean_area_um2': aggregation_index
            }
            
        except Exception as e:
            print(f"Error analyzing particle aggregation for {image_path}: {e}")
            return {}
    
    def analyze_porosity(self, image_path: Path) -> Dict:
        """
        Analyze pore structure and void spaces.
        
        Args:
            image_path: Path to the SEM image
            
        Returns:
            Dictionary containing porosity metrics
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {}
            
            # Invert image to make pores white
            img_inv = 255 - img
            
            # Threshold to identify pores
            _, pore_binary = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Clean up pore detection
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            pore_binary = cv2.morphologyEx(pore_binary, cv2.MORPH_OPEN, kernel)
            
            # Calculate porosity metrics
            total_image_area = img.shape[0] * img.shape[1]
            total_pore_area = np.sum(pore_binary > 0)
            porosity = total_pore_area / total_image_area
            
            # Find individual pores
            contours, _ = cv2.findContours(pore_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pore_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 5]
            
            if pore_areas:
                mean_pore_size = np.mean(pore_areas)
                pore_size_std = np.std(pore_areas)
                pore_count = len(pore_areas)
            else:
                mean_pore_size = 0
                pore_size_std = 0
                pore_count = 0
            
            return {
                'porosity': porosity,
                'pore_count': pore_count,
                'mean_pore_size': mean_pore_size,
                'pore_size_std': pore_size_std,
                'pore_density': pore_count / total_image_area * 1e6
            }
            
        except Exception as e:
            print(f"Error analyzing porosity for {image_path}: {e}")
            return {}
    
    def analyze_boundary_definition(self, image_path: Path) -> Dict:
        """
        Analyze particle boundary definition and clarity.
        
        Args:
            image_path: Path to the SEM image
            
        Returns:
            Dictionary containing boundary metrics
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {}
            
            # Edge detection methods
            edges_canny = cv2.Canny(img, self.config['canny_threshold1'], self.config['canny_threshold2'])
            
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            edges_sobel = np.sqrt(sobelx**2 + sobely**2)
            
            # Boundary metrics
            boundary_sharpness = np.std(edges_canny)
            boundary_continuity = np.sum(edges_canny > 0) / edges_canny.size
            mean_gradient = np.mean(np.sqrt(sobelx**2 + sobely**2))
            gradient_std = np.std(np.sqrt(sobelx**2 + sobely**2))
            
            return {
                'boundary_sharpness': boundary_sharpness,
                'boundary_continuity': boundary_continuity,
                'mean_gradient': mean_gradient,
                'gradient_std': gradient_std,
                'edge_strength': np.mean(edges_sobel)
            }
            
        except Exception as e:
            print(f"Error analyzing boundaries for {image_path}: {e}")
            return {}
    
    def analyze_image(self, image_path: Path, condition: str = "", metadata: Dict = None) -> Dict:
        """
        Perform comprehensive analysis on a single SEM image.
        
        Args:
            image_path: Path to the SEM image
            condition: Experimental condition label
            metadata: Additional metadata dictionary
            
        Returns:
            Dictionary containing all analysis results
        """
        print(f"Analyzing {image_path.name}...")
        
        # Run all analysis types
        roughness_metrics = self.analyze_surface_roughness(image_path)
        aggregation_metrics = self.analyze_particle_aggregation(image_path)
        porosity_metrics = self.analyze_porosity(image_path)
        boundary_metrics = self.analyze_boundary_definition(image_path)
        
        # Combine results
        result = {
            'image_file': image_path.name,
            'condition': condition,
            'image_path': str(image_path),
            **roughness_metrics,
            **aggregation_metrics,
            **porosity_metrics,
            **boundary_metrics
        }
        
        # Add metadata if provided
        if metadata:
            result.update(metadata)
        
        return result
    
    def analyze_dataset(self, input_dir: Path, condition_mapping: Dict = None) -> pd.DataFrame:
        """
        Analyze a complete dataset of SEM images.
        
        Args:
            input_dir: Directory containing SEM images
            condition_mapping: Dictionary mapping subfolder names to conditions
            
        Returns:
            DataFrame containing all analysis results
        """
        results = []
        
        # If condition mapping provided, analyze by subdirectories
        if condition_mapping:
            for subfolder, condition in condition_mapping.items():
                subfolder_path = input_dir / subfolder
                if subfolder_path.exists():
                    print(f"Processing condition: {condition}")
                    for img_file in subfolder_path.glob("*.tif"):
                        result = self.analyze_image(img_file, condition)
                        results.append(result)
        else:
            # Analyze all images in the main directory
            for img_file in input_dir.glob("*.tif"):
                result = self.analyze_image(img_file)
                results.append(result)
        
        return pd.DataFrame(results)
    
    def create_visualizations(self, df: pd.DataFrame, output_dir: Path):
        """
        Create comprehensive visualizations of the analysis results.
        
        Args:
            df: DataFrame containing analysis results
            output_dir: Directory to save visualizations
        """
        output_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Check available columns for plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        plot_cols = [col for col in numeric_cols if col not in ['particle_count', 'pore_count']]
        
        if len(plot_cols) == 0:
            print("No suitable columns for visualization")
            return
        
        # Main analysis figure
        n_plots = min(6, len(plot_cols))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SEM Surface Morphology Analysis Results', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, col in enumerate(plot_cols[:n_plots]):
            if 'condition' in df.columns and len(df['condition'].unique()) > 1:
                sns.boxplot(data=df, x='condition', y=col, ax=axes[i])
                axes[i].tick_params(axis='x', rotation=45)
            else:
                df[col].hist(bins=20, ax=axes[i])
            axes[i].set_title(col.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_plots, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'morphology_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation matrix if multiple metrics available
        if len(plot_cols) > 3:
            plt.figure(figsize=(12, 10))
            correlation_matrix = df[plot_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Morphological Parameter Correlations')
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_report(self, df: pd.DataFrame, output_dir: Path):
        """
        Generate a comprehensive analysis report.
        
        Args:
            df: DataFrame containing analysis results
            output_dir: Directory to save the report
        """
        output_dir.mkdir(exist_ok=True)
        
        report_path = output_dir / 'analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# SEM Surface Morphology Analysis Report\n\n")
            f.write(f"## Analysis Summary\n\n")
            f.write(f"- **Total images analyzed**: {len(df)}\n")
            f.write(f"- **Analysis date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if 'condition' in df.columns:
                f.write(f"- **Experimental conditions**: {', '.join(df['condition'].unique())}\n")
            
            f.write("\n## Statistical Summary\n\n")
            
            # Generate summary statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            summary_stats = df[numeric_cols].describe()
            
            f.write("| Parameter | Mean | Std | Min | Max |\n")
            f.write("|-----------|------|-----|-----|-----|\n")
            
            for param in numeric_cols[:10]:  # Limit to first 10 parameters
                row = summary_stats.loc[:, param]
                f.write(f"| {param} | {row['mean']:.3f} | {row['std']:.3f} | {row['min']:.3f} | {row['max']:.3f} |\n")
            
            f.write("\n## Methodology\n\n")
            f.write("This analysis was performed using the SEM Surface Morphology Analyzer toolkit.\n")
            f.write("The following parameters were extracted:\n\n")
            f.write("- **Surface roughness**: Texture contrast, homogeneity, surface variation\n")
            f.write("- **Particle aggregation**: Size heterogeneity, aggregation index, particle density\n")
            f.write("- **Porosity**: Total porosity, pore count, pore size distribution\n")
            f.write("- **Boundary definition**: Boundary sharpness, continuity, edge density\n\n")
            
            f.write("## Configuration Used\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.config, indent=2))
            f.write("\n```\n")
        
        print(f"Report generated: {report_path}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='SEM Surface Morphology Analyzer')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing SEM images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save analysis results')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to JSON configuration file')
    parser.add_argument('--condition_mapping', type=str, default=None,
                       help='Path to JSON file mapping subfolders to conditions')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Load condition mapping if provided
    condition_mapping = None
    if args.condition_mapping:
        with open(args.condition_mapping, 'r') as f:
            condition_mapping = json.load(f)
    
    # Initialize analyzer
    analyzer = SEMMorphologyAnalyzer(config)
    
    # Run analysis
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("Starting SEM morphology analysis...")
    df = analyzer.analyze_dataset(input_dir, condition_mapping)
    
    if df.empty:
        print("No data collected. Check input directory and image files.")
        return
    
    # Save results
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / 'morphology_analysis_results.csv', index=False)
    
    # Create visualizations
    analyzer.create_visualizations(df, output_dir)
    
    # Generate report
    analyzer.generate_report(df, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
