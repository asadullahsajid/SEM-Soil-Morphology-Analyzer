"""
Example usage script for SEM Surface Morphology Analyzer
This script demonstrates how to use the toolkit for basic analysis
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sem_morphology_analyzer import SEMMorphologyAnalyzer

def main():
    """
    Example usage of the SEM Surface Morphology Analyzer
    """
    print("SEM Surface Morphology Analyzer - Example Usage")
    print("=" * 50)
    
    # Example 1: Basic single image analysis
    print("\n1. Single Image Analysis Example:")
    
    # Initialize analyzer with default configuration
    analyzer = SEMMorphologyAnalyzer()
    
    # Example image path (replace with your actual image)
    # image_path = Path("path/to/your/image.tif")
    
    # Uncomment the following lines when you have an actual image:
    # result = analyzer.analyze_image(image_path, condition="example_condition")
    # print(f"Analysis complete for {image_path.name}")
    # print(f"Surface variation: {result.get('surface_variation', 'N/A')}")
    # print(f"Particle count: {result.get('particle_count', 'N/A')}")
    
    print("Single image analysis example (commented out - add your image path)")
    
    # Example 2: Custom configuration
    print("\n2. Custom Configuration Example:")
    
    custom_config = {
        'scale_pixels_per_um': 150,  # Higher magnification
        'min_particle_area_um2': 0.005,  # Smaller minimum particle size
        'canny_threshold1': 30,  # Lower edge detection threshold
        'canny_threshold2': 100
    }
    
    custom_analyzer = SEMMorphologyAnalyzer(custom_config)
    print("Custom analyzer created with modified parameters:")
    print(f"- Scale: {custom_config['scale_pixels_per_um']} pixels/μm")
    print(f"- Min particle area: {custom_config['min_particle_area_um2']} μm²")
    
    # Example 3: Dataset analysis
    print("\n3. Dataset Analysis Example:")
    
    # Example directory structure:
    # input_directory/
    #   ├── condition_1/
    #   │   ├── image1.tif
    #   │   └── image2.tif
    #   └── condition_2/
    #       ├── image3.tif
    #       └── image4.tif
    
    condition_mapping = {
        "before_treatment": "Control",
        "after_treatment": "Treated"
    }
    
    # Uncomment when you have actual data:
    # input_dir = Path("path/to/your/dataset")
    # df = analyzer.analyze_dataset(input_dir, condition_mapping)
    # output_dir = Path("path/to/output")
    # analyzer.create_visualizations(df, output_dir)
    # analyzer.generate_report(df, output_dir)
    
    print("Dataset analysis example (commented out - add your data paths)")
    
    print("\n" + "=" * 50)
    print("Example complete! Uncomment sections with actual image paths to run analysis.")

if __name__ == "__main__":
    main()
