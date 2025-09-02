"""
Unit tests for SEM Surface Morphology Analyzer
Run with: python -m pytest tests/
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / "src"))

from sem_morphology_analyzer import SEMMorphologyAnalyzer
class TestSEMMorphologyAnalyzer:
    """Test cases for the SEM Surface Morphology Analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create a test analyzer instance"""
        return SEMMorphologyAnalyzer()
    
    @pytest.fixture
    def test_image(self):
        """Create a synthetic test image"""
        # Create a simple test image with some patterns
        img = np.zeros((512, 512), dtype=np.uint8)
        
        # Add some circular "particles"
        cv2.circle(img, (100, 100), 20, 255, -1)
        cv2.circle(img, (200, 200), 30, 255, -1)
        cv2.circle(img, (350, 350), 25, 255, -1)
        
        # Add some noise for texture
        noise = np.random.randint(0, 50, (512, 512), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly"""
        assert analyzer is not None
        assert analyzer.config is not None
        assert 'scale_pixels_per_um' in analyzer.config
    
    def test_custom_config(self):
        """Test analyzer with custom configuration"""
        custom_config = {'scale_pixels_per_um': 150}
        analyzer = SEMMorphologyAnalyzer(custom_config)
        assert analyzer.config['scale_pixels_per_um'] == 150
    
    def test_surface_roughness_analysis(self, analyzer, test_image):
        """Test surface roughness analysis function"""
        # Save test image temporarily
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
            
            # Run analysis
            result = analyzer.analyze_surface_roughness(tmp_path)
            
            # Clean up
            tmp_path.unlink()
            
            # Check results
            assert isinstance(result, dict)
            assert 'surface_variation' in result
            assert 'texture_contrast' in result
            assert result['surface_variation'] > 0
    
    def test_particle_aggregation_analysis(self, analyzer, test_image):
        """Test particle aggregation analysis function"""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
            
            result = analyzer.analyze_particle_aggregation(tmp_path)
            tmp_path.unlink()
            
            assert isinstance(result, dict)
            assert 'particle_count' in result
            assert 'aggregation_index' in result
            # Should detect some particles in our synthetic image
            assert result['particle_count'] > 0
    
    def test_porosity_analysis(self, analyzer, test_image):
        """Test porosity analysis function"""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
            
            result = analyzer.analyze_porosity(tmp_path)
            tmp_path.unlink()
            
            assert isinstance(result, dict)
            assert 'porosity' in result
            assert 'pore_count' in result
            assert 0 <= result['porosity'] <= 1
    
    def test_boundary_analysis(self, analyzer, test_image):
        """Test boundary definition analysis function"""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
            
            result = analyzer.analyze_boundary_definition(tmp_path)
            tmp_path.unlink()
            
            assert isinstance(result, dict)
            assert 'boundary_sharpness' in result
            assert 'boundary_continuity' in result
            assert result['boundary_sharpness'] >= 0
    
    def test_comprehensive_analysis(self, analyzer, test_image):
        """Test the complete analysis pipeline"""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            tmp_path = Path(tmp.name)
            
            result = analyzer.analyze_image(tmp_path, condition="test")
            tmp_path.unlink()
            
            # Check that all analysis types are included
            assert 'condition' in result
            assert 'image_file' in result
            assert 'surface_variation' in result
            assert 'particle_count' in result
            assert 'porosity' in result
            assert 'boundary_sharpness' in result
    
    def test_invalid_image_handling(self, analyzer):
        """Test handling of invalid image paths"""
        invalid_path = Path("nonexistent_image.tif")
        
        # Should return empty dict for invalid images
        result = analyzer.analyze_surface_roughness(invalid_path)
        assert result == {}
        
        result = analyzer.analyze_particle_aggregation(invalid_path)
        assert result == {}

if __name__ == "__main__":
    pytest.main([__file__])
