#!/usr/bin/env python3
"""
Quick test script to verify the SEM Morphology Analyzer is working correctly.

This script performs a basic test of the analyzer functionality using the sample data.
Run this to verify your installation is working before uploading to GitHub.

Author: Hafiz Asad Ullah Sajid
Email: hafizasadullahsajid.iub@gmail.com
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_installation():
    """Test that all components are working correctly."""
    
    print("🔍 SEM Morphology Analyzer - Installation Test")
    print("=" * 50)
    
    # Test 1: Import main module
    try:
        from sem_morphology_analyzer import SEMMorphologyAnalyzer
        print("✅ Successfully imported SEMMorphologyAnalyzer")
    except Exception as e:
        print(f"❌ Failed to import analyzer: {e}")
        return False
    
    # Test 2: Create analyzer instance
    try:
        analyzer = SEMMorphologyAnalyzer()
        print("✅ Successfully created analyzer instance")
    except Exception as e:
        print(f"❌ Failed to create analyzer: {e}")
        return False
    
    # Test 3: Check sample data
    sample_files = [
        'sample_data/soil_1/before_washing.tif',
        'sample_data/soil_1/after_nacetyl.tif',
        'sample_data/soil_1/after_oxalic.tif',
        'sample_data/soil_2/before_washing.tif',
        'sample_data/soil_2/after_nacetyl.tif',
        'sample_data/soil_2/after_oxalic.tif'
    ]
    
    missing_files = []
    for file_path in sample_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing sample files: {missing_files}")
        return False
    else:
        print("✅ All sample data files are present")
    
    # Test 4: Quick analysis test
    try:
        test_image = 'sample_data/soil_1/before_washing.tif'
        print(f"🔬 Testing analysis on: {test_image}")
        
        # Run a quick surface roughness analysis
        result = analyzer.analyze_surface_roughness(test_image)
        
        if result and 'surface_variation' in result:
            print(f"✅ Analysis successful! Surface variation: {result['surface_variation']:.4f}")
        else:
            print("❌ Analysis failed - no results returned")
            return False
            
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        return False
    
    print("\n🎉 All tests passed! Your SEM Morphology Analyzer is ready to use.")
    print("\nNext steps:")
    print("1. Upload your repository to GitHub")
    print("2. Try running: python examples/analyze_sample_data.py")
    print("3. Use the toolkit for your own SEM images")
    
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)
