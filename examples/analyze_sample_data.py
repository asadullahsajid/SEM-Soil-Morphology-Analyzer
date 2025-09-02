#!/usr/bin/env python3
"""
Example script demonstrating SEM morphology analysis using the included sample data.

This script analyzes the sample SEM images included in the repository
to demonstrate the capabilities of the SEM Morphology Analyzer toolkit.

Author: Hafiz Asad Ullah Sajid
Institution: Sichuan Agricultural University, Department of Environmental Engineering
Email: hafizasadullahsajid.iub@gmail.com
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import pandas as pd

# Add the src directory to the path to import our analyzer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sem_morphology_analyzer import SEMMorphologyAnalyzer

def main():
    """
    Analyze the sample SEM images included in the repository.
    """
    print("SEM Morphology Analysis - Sample Data Demonstration")
    print("=" * 60)
    print("Analyzing sample SEM images from soil washing research")
    print("Treatment conditions: Before washing, N-Acetyl L-Glutamic acid, Oxalic acid")
    print()
    
    # Define paths to sample data
    sample_data_dir = os.path.join(os.path.dirname(__file__), '..', 'sample_data')
    results_dir = os.path.join(os.path.dirname(__file__), 'sample_results')
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize the analyzer
    analyzer = SEMMorphologyAnalyzer()
    
    # Define the sample images to analyze
    sample_images = {
        'Soil Type 1 - Before Washing': os.path.join(sample_data_dir, 'soil_1', 'before_washing.tif'),
        'Soil Type 1 - After N-Acetyl': os.path.join(sample_data_dir, 'soil_1', 'after_nacetyl.tif'),
        'Soil Type 1 - After Oxalic': os.path.join(sample_data_dir, 'soil_1', 'after_oxalic.tif'),
        'Soil Type 2 - Before Washing': os.path.join(sample_data_dir, 'soil_2', 'before_washing.tif'),
        'Soil Type 2 - After N-Acetyl': os.path.join(sample_data_dir, 'soil_2', 'after_nacetyl.tif'),
        'Soil Type 2 - After Oxalic': os.path.join(sample_data_dir, 'soil_2', 'after_oxalic.tif'),
    }
    
    # Results storage
    all_results = {}
    
    # Analyze each sample image
    for name, image_path in sample_images.items():
        if os.path.exists(image_path):
            print(f"Analyzing: {name}")
            try:
                # Perform comprehensive analysis
                results = analyzer.analyze_image(image_path)
                all_results[name] = results
                
                # Save individual results
                result_file = os.path.join(results_dir, f"{name.replace(' ', '_').replace('-', '_').lower()}_results.json")
                with open(result_file, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                
                print(f"  - Particles detected: {results['particle_count']}")
                print(f"  - Mean particle size: {results['mean_particle_size']:.2f} μm")
                print(f"  - Surface roughness: {results['surface_roughness']:.4f}")
                print(f"  - Porosity index: {results['porosity_index']:.2f}%")
                print()
                
            except Exception as e:
                print(f"  - Error analyzing {name}: {str(e)}")
                continue
        else:
            print(f"Warning: Sample image not found: {image_path}")
    
    # Generate comparative analysis
    if len(all_results) > 1:
        print("Generating comparative analysis...")
        generate_comparative_plots(all_results, results_dir)
        save_summary_table(all_results, results_dir)
    
    print(f"\nAnalysis complete! Results saved in: {results_dir}")
    print("Files generated:")
    print("- Individual JSON results for each image")
    print("- summary_table.csv: Comparative metrics")
    print("- comparative_analysis.png: Visual comparison")

def generate_comparative_plots(results, output_dir):
    """Generate comparative plots for the analysis results."""
    
    # Extract data for plotting
    names = list(results.keys())
    particle_counts = [results[name]['particle_count'] for name in names]
    mean_sizes = [results[name]['mean_particle_size'] for name in names]
    roughness = [results[name]['surface_roughness'] for name in names]
    porosity = [results[name]['porosity_index'] for name in names]
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SEM Morphology Analysis - Sample Data Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Particle counts
    axes[0, 0].bar(range(len(names)), particle_counts, color='skyblue')
    axes[0, 0].set_title('Particle Count by Treatment')
    axes[0, 0].set_ylabel('Number of Particles')
    axes[0, 0].set_xticks(range(len(names)))
    axes[0, 0].set_xticklabels([name.replace(' - ', '\n') for name in names], rotation=45, ha='right')
    
    # Plot 2: Mean particle sizes
    axes[0, 1].bar(range(len(names)), mean_sizes, color='lightcoral')
    axes[0, 1].set_title('Mean Particle Size by Treatment')
    axes[0, 1].set_ylabel('Mean Size (μm)')
    axes[0, 1].set_xticks(range(len(names)))
    axes[0, 1].set_xticklabels([name.replace(' - ', '\n') for name in names], rotation=45, ha='right')
    
    # Plot 3: Surface roughness
    axes[1, 0].bar(range(len(names)), roughness, color='lightgreen')
    axes[1, 0].set_title('Surface Roughness by Treatment')
    axes[1, 0].set_ylabel('Roughness Index')
    axes[1, 0].set_xticks(range(len(names)))
    axes[1, 0].set_xticklabels([name.replace(' - ', '\n') for name in names], rotation=45, ha='right')
    
    # Plot 4: Porosity
    axes[1, 1].bar(range(len(names)), porosity, color='gold')
    axes[1, 1].set_title('Porosity Index by Treatment')
    axes[1, 1].set_ylabel('Porosity (%)')
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_xticklabels([name.replace(' - ', '\n') for name in names], rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparative_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_summary_table(results, output_dir):
    """Save a summary table of all results."""
    
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Sample': name,
            'Particle_Count': result['particle_count'],
            'Mean_Size_um': result['mean_particle_size'],
            'Std_Size_um': result['std_particle_size'],
            'Min_Size_um': result['min_particle_size'],
            'Max_Size_um': result['max_particle_size'],
            'Surface_Roughness': result['surface_roughness'],
            'Porosity_Index_percent': result['porosity_index'],
            'Aggregation_Index': result['aggregation_index'],
            'Shape_Complexity': result['shape_complexity']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'summary_table.csv'), index=False)
    
    # Also save a formatted version for easy reading
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("SEM Morphology Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Analysis conducted using SEM Morphology Analyzer\n")
        f.write("Author: Hafiz Asad Ullah Sajid\n")
        f.write("Institution: Sichuan Agricultural University\n\n")
        
        for _, row in df.iterrows():
            f.write(f"Sample: {row['Sample']}\n")
            f.write(f"  Particles detected: {row['Particle_Count']}\n")
            f.write(f"  Mean size: {row['Mean_Size_um']:.2f} ± {row['Std_Size_um']:.2f} μm\n")
            f.write(f"  Size range: {row['Min_Size_um']:.2f} - {row['Max_Size_um']:.2f} μm\n")
            f.write(f"  Surface roughness: {row['Surface_Roughness']:.4f}\n")
            f.write(f"  Porosity index: {row['Porosity_Index_percent']:.2f}%\n")
            f.write(f"  Aggregation index: {row['Aggregation_Index']:.3f}\n")
            f.write(f"  Shape complexity: {row['Shape_Complexity']:.3f}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
