"""
Comprehensive test suite for HeatmapGenerator module.

This test suite validates all functionality of the dual-resolution heatmap generation
system, including coordinate mapping, pattern generation, and feature extraction.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.heatmap_generator import (
    HeatmapGenerator, 
    ResolutionConfig, 
    PatternType, 
    HeatmapData
)


class TestHeatmapGenerator:
    """Test class for HeatmapGenerator functionality."""
    
    def __init__(self):
        """Initialize test environment."""
        self.generator = HeatmapGenerator()
        self.results = []
        print("🧪 HeatmapGenerator Test Suite")
        print("=" * 50)
    
    def test_initialization(self):
        """Test generator initialization and configuration."""
        print("\n📋 Testing Initialization...")
        
        # Test default configuration
        assert self.generator.config.display_size == (1000, 1000)
        assert self.generator.config.grid_size == (50, 50)
        assert self.generator.scale_factor == (20, 20)
        print("✅ Default configuration correct")
        
        # Test custom configuration
        custom_config = ResolutionConfig(
            display_size=(800, 600),
            grid_size=(40, 30)
        )
        custom_generator = HeatmapGenerator(custom_config)
        assert custom_generator.scale_factor == (20, 20)
        print("✅ Custom configuration correct")
        
        # Test invalid configuration
        try:
            invalid_config = ResolutionConfig(
                display_size=(1000, 1000),
                grid_size=(33, 33)  # Not evenly divisible
            )
            HeatmapGenerator(invalid_config)
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✅ Invalid configuration properly rejected")
    
    def test_coordinate_mapping(self):
        """Test bidirectional coordinate mapping."""
        print("\n🗺️ Testing Coordinate Mapping...")
        
        # Test grid to display mapping
        grid_coords = [(0, 0), (1, 1), (25, 25), (49, 49)]
        expected_display = [(0, 0), (20, 20), (500, 500), (980, 980)]
        
        for i, (gr, gc) in enumerate(grid_coords):
            dr, dc = self.generator.grid_to_display_coords(gr, gc)
            expected = expected_display[i]
            assert (dr, dc) == expected, f"Expected {expected}, got {(dr, dc)}"
        print("✅ Grid to display mapping correct")
        
        # Test display to grid mapping
        display_coords = [(0, 0), (20, 20), (500, 500), (980, 980), (999, 999)]
        expected_grid = [(0, 0), (1, 1), (25, 25), (49, 49), (49, 49)]
        
        for i, (dr, dc) in enumerate(display_coords):
            gr, gc = self.generator.display_to_grid_coords(dr, dc)
            expected = expected_grid[i]
            assert (gr, gc) == expected, f"Expected {expected}, got {(gr, gc)}"
        print("✅ Display to grid mapping correct")
        
        # Test round-trip accuracy
        for gr in range(0, 50, 10):
            for gc in range(0, 50, 10):
                dr, dc = self.generator.grid_to_display_coords(gr, gc)
                back_gr, back_gc = self.generator.display_to_grid_coords(dr, dc)
                assert (gr, gc) == (back_gr, back_gc), f"Round-trip failed for {(gr, gc)}"
        print("✅ Round-trip mapping accuracy verified")
    
    def test_gaussian_hotspot_generation(self):
        """Test Gaussian hotspot pattern generation."""
        print("\n🔥 Testing Gaussian Hotspot Generation...")
        
        # Test default parameters
        heatmap_data = self.generator.generate_heatmap(PatternType.GAUSSIAN_HOTSPOT)
        
        # Verify data structure
        assert isinstance(heatmap_data, HeatmapData)
        assert heatmap_data.display_layer.shape == (1000, 1000)
        assert heatmap_data.decision_grid.shape == (50, 50)
        assert 'buildings' in heatmap_data.features
        print("✅ Default Gaussian hotspot structure correct")
        
        # Test custom parameters
        custom_centers = [(200, 200), (800, 800)]
        custom_intensities = [1.0, 0.5]
        custom_sigma = 50
        
        custom_heatmap = self.generator.generate_heatmap(
            PatternType.GAUSSIAN_HOTSPOT,
            hotspot_centers=custom_centers,
            intensities=custom_intensities,
            sigma=custom_sigma
        )
        
        # Verify hotspots are at expected locations (approximately)
        display_layer = custom_heatmap.display_layer
        
        # Check first hotspot (should be brightest)
        region1 = display_layer[180:220, 180:220]
        max1_pos = np.unravel_index(np.argmax(region1), region1.shape)
        assert abs(max1_pos[0] - 20) < 5 and abs(max1_pos[1] - 20) < 5
        
        # Check second hotspot
        region2 = display_layer[780:820, 780:820]
        max2_pos = np.unravel_index(np.argmax(region2), region2.shape)
        assert abs(max2_pos[0] - 20) < 5 and abs(max2_pos[1] - 20) < 5
        
        print("✅ Custom Gaussian hotspot parameters working")
        
        # Test intensity validation
        peak1 = np.max(display_layer[180:220, 180:220])
        peak2 = np.max(display_layer[780:820, 780:820])
        assert peak1 > peak2, f"First hotspot should be brighter: {peak1} vs {peak2}"
        print("✅ Intensity differences correctly applied")
        
        # Store for visualization
        self.results.append(("Gaussian Hotspot", custom_heatmap))
    
    def test_horizontal_gradient_generation(self):
        """Test horizontal gradient pattern generation."""
        print("\n➡️ Testing Horizontal Gradient Generation...")
        
        # Test left to right gradient
        left_right_heatmap = self.generator.generate_heatmap(
            PatternType.HORIZONTAL_GRADIENT,
            direction='left_to_right',
            min_value=0.2,
            max_value=0.8
        )
        
        display_layer = left_right_heatmap.display_layer
        
        # Verify gradient direction and values
        assert abs(display_layer[500, 0] - 0.2) < 0.01, f"Left edge should be ~0.2, got {display_layer[500, 0]}"
        assert abs(display_layer[500, 999] - 0.8) < 0.01, f"Right edge should be ~0.8, got {display_layer[500, 999]}"
        assert display_layer[500, 100] < display_layer[500, 900], "Should increase from left to right"
        print("✅ Left-to-right gradient correct")
        
        # Test right to left gradient
        right_left_heatmap = self.generator.generate_heatmap(
            PatternType.HORIZONTAL_GRADIENT,
            direction='right_to_left',
            min_value=0.1,
            max_value=0.9
        )
        
        display_layer = right_left_heatmap.display_layer
        assert abs(display_layer[500, 0] - 0.9) < 0.01, "Left edge should be max value"
        assert abs(display_layer[500, 999] - 0.1) < 0.01, "Right edge should be min value"
        assert display_layer[500, 100] > display_layer[500, 900], "Should decrease from left to right"
        print("✅ Right-to-left gradient correct")
        
        # Test horizontal consistency (all rows should be identical)
        for row in [0, 250, 500, 750, 999]:
            np.testing.assert_array_almost_equal(
                display_layer[row, :], 
                display_layer[0, :], 
                decimal=6,
                err_msg=f"Row {row} should be identical to row 0"
            )
        print("✅ Horizontal consistency verified")
        
        # Store for visualization
        self.results.append(("Horizontal Gradient L→R", left_right_heatmap))
        self.results.append(("Horizontal Gradient R→L", right_left_heatmap))
    
    def test_vertical_gradient_generation(self):
        """Test vertical gradient pattern generation."""
        print("\n⬇️ Testing Vertical Gradient Generation...")
        
        # Test top to bottom gradient
        top_bottom_heatmap = self.generator.generate_heatmap(
            PatternType.VERTICAL_GRADIENT,
            direction='top_to_bottom',
            min_value=0.3,
            max_value=0.7
        )
        
        display_layer = top_bottom_heatmap.display_layer
        
        # Verify gradient direction and values
        assert abs(display_layer[0, 500] - 0.3) < 0.01, f"Top edge should be ~0.3, got {display_layer[0, 500]}"
        assert abs(display_layer[999, 500] - 0.7) < 0.01, f"Bottom edge should be ~0.7, got {display_layer[999, 500]}"
        assert display_layer[100, 500] < display_layer[900, 500], "Should increase from top to bottom"
        print("✅ Top-to-bottom gradient correct")
        
        # Test bottom to top gradient
        bottom_top_heatmap = self.generator.generate_heatmap(
            PatternType.VERTICAL_GRADIENT,
            direction='bottom_to_top',
            min_value=0.2,
            max_value=0.8
        )
        
        display_layer = bottom_top_heatmap.display_layer
        assert abs(display_layer[0, 500] - 0.8) < 0.01, "Top edge should be max value"
        assert abs(display_layer[999, 500] - 0.2) < 0.01, "Bottom edge should be min value"
        assert display_layer[100, 500] > display_layer[900, 500], "Should decrease from top to bottom"
        print("✅ Bottom-to-top gradient correct")
        
        # Test vertical consistency (all columns should be identical)
        for col in [0, 250, 500, 750, 999]:
            np.testing.assert_array_almost_equal(
                display_layer[:, col], 
                display_layer[:, 0], 
                decimal=6,
                err_msg=f"Column {col} should be identical to column 0"
            )
        print("✅ Vertical consistency verified")
        
        # Store for visualization
        self.results.append(("Vertical Gradient T→B", top_bottom_heatmap))
        self.results.append(("Vertical Gradient B→T", bottom_top_heatmap))
    
    def test_radial_gradient_generation(self):
        """Test radial gradient pattern generation."""
        print("\n🎯 Testing Radial Gradient Generation...")
        
        # Test center outward gradient
        center_out_heatmap = self.generator.generate_heatmap(
            PatternType.RADIAL_GRADIENT,
            direction='center_outward',
            center=(500, 500),
            min_value=0.2,
            max_value=0.8,
            max_radius=400
        )
        
        display_layer = center_out_heatmap.display_layer
        
        # Verify gradient direction and values
        center_value = display_layer[500, 500]  # Center should be min_value
        edge_value = display_layer[100, 500]    # Far from center should be higher
        corner_value = display_layer[0, 0]      # Corner should be max or clamped
        
        assert abs(center_value - 0.2) < 0.05, f"Center should be ~0.2, got {center_value}"
        assert center_value < edge_value, f"Center ({center_value}) should be less than edge ({edge_value})"
        print("✅ Center-outward gradient correct")
        
        # Test outward center gradient  
        out_center_heatmap = self.generator.generate_heatmap(
            PatternType.RADIAL_GRADIENT,
            direction='outward_center',
            center=(500, 500),
            min_value=0.1,
            max_value=0.9,
            max_radius=300
        )
        
        display_layer = out_center_heatmap.display_layer
        center_value = display_layer[500, 500]  # Center should be max_value
        edge_value = display_layer[200, 500]    # Far from center should be lower
        
        assert abs(center_value - 0.9) < 0.05, f"Center should be ~0.9, got {center_value}"
        assert center_value > edge_value, f"Center ({center_value}) should be greater than edge ({edge_value})"
        print("✅ Outward-center gradient correct")
        
        # Test radial symmetry
        center_y, center_x = 500, 500
        test_points = [
            (center_y - 100, center_x),      # North
            (center_y + 100, center_x),      # South  
            (center_y, center_x - 100),      # West
            (center_y, center_x + 100),      # East
        ]
        
        center_out_layer = center_out_heatmap.display_layer
        values = [center_out_layer[y, x] for y, x in test_points]
        
        # All points equidistant from center should have similar values
        for i in range(1, len(values)):
            assert abs(values[0] - values[i]) < 0.05, f"Radial symmetry broken: {values}"
        print("✅ Radial symmetry verified")
        
        # Test custom center position
        custom_center_heatmap = self.generator.generate_heatmap(
            PatternType.RADIAL_GRADIENT,
            direction='center_outward',
            center=(300, 700),  # Off-center
            min_value=0.0,
            max_value=1.0,
            max_radius=200
        )
        
        display_layer = custom_center_heatmap.display_layer
        custom_center_value = display_layer[300, 700]
        nearby_value = display_layer[350, 700]  # 50 pixels away
        
        assert abs(custom_center_value - 0.0) < 0.05, f"Custom center should be ~0.0, got {custom_center_value}"
        assert custom_center_value < nearby_value, "Should increase away from custom center"
        print("✅ Custom center position working")
        
        # Store for visualization
        self.results.append(("Radial Center→Out", center_out_heatmap))
        self.results.append(("Radial Out→Center", out_center_heatmap))
        self.results.append(("Radial Custom Center", custom_center_heatmap))
    
    def test_downsampling_accuracy(self):
        """Test display layer to decision grid downsampling."""
        print("\n⬇️ Testing Downsampling Accuracy...")
        
        # Create a simple test pattern with known values
        test_display = np.zeros((1000, 1000))
        # Fill specific grid cells with known values
        test_display[0:20, 0:20] = 1.0    # Grid cell (0,0) = 1.0
        test_display[20:40, 20:40] = 0.5  # Grid cell (1,1) = 0.5
        test_display[980:1000, 980:1000] = 0.25  # Grid cell (49,49) = 0.25
        
        # Test downsampling
        downsampled = self.generator._downsample_to_grid(test_display)
        
        # Verify results
        assert abs(downsampled[0, 0] - 1.0) < 1e-6, f"Expected 1.0, got {downsampled[0, 0]}"
        assert abs(downsampled[1, 1] - 0.5) < 1e-6, f"Expected 0.5, got {downsampled[1, 1]}"
        assert abs(downsampled[49, 49] - 0.25) < 1e-6, f"Expected 0.25, got {downsampled[49, 49]}"
        
        print("✅ Downsampling preserves grid cell averages")
        
        # Test gradient preservation
        gradient_display = np.linspace(0, 1, 1000).reshape(1, 1000)
        gradient_display = np.repeat(gradient_display, 1000, axis=0)
        
        gradient_downsampled = self.generator._downsample_to_grid(gradient_display)
        
        # Should have smooth gradient in decision grid too
        assert gradient_downsampled[0, 0] < gradient_downsampled[0, 25] < gradient_downsampled[0, 49]
        print("✅ Gradient patterns preserved during downsampling")
    
    def test_feature_extraction(self):
        """Test feature extraction from generated heatmaps."""
        print("\n🔍 Testing Feature Extraction...")
        
        # Generate test heatmap
        heatmap_data = self.generator.generate_heatmap(PatternType.GAUSSIAN_HOTSPOT)
        
        # Test feature structure
        features = heatmap_data.features
        assert 'buildings' in features
        assert 'obstacles' in features
        assert 'sensor_areas' in features
        print("✅ All expected features present")
        
        # Test feature dimensions
        assert features['buildings'].shape == (1000, 1000)
        assert features['obstacles'].shape == (1000, 1000)
        assert features['sensor_areas'].shape == (50, 50)
        print("✅ Feature dimensions correct")
        
        # Test feature types
        assert features['buildings'].dtype == np.uint8
        assert features['obstacles'].dtype == np.uint8
        assert features['sensor_areas'].dtype == bool
        print("✅ Feature data types correct")
        
        # Test that features identify something
        assert np.sum(features['buildings']) > 0, "Buildings should be detected"
        assert np.sum(features['obstacles']) > 0, "Obstacles should be detected"
        assert np.sum(features['sensor_areas']) > 0, "Sensor areas should be valid"
        print("✅ Features identify expected regions")
    
    def test_save_load_functionality(self):
        """Test saving and loading heatmap data."""
        print("\n💾 Testing Save/Load Functionality...")
        
        # Generate test data
        original_data = self.generator.generate_heatmap(PatternType.GAUSSIAN_HOTSPOT)
        
        # Save data
        test_file = "test_heatmap_save.npz"
        self.generator.save_heatmap(original_data, test_file)
        
        # Load data
        loaded_data = HeatmapGenerator.load_heatmap(test_file)
        
        # Verify loaded data matches original
        np.testing.assert_array_equal(original_data.display_layer, loaded_data.display_layer)
        np.testing.assert_array_equal(original_data.decision_grid, loaded_data.decision_grid)
        np.testing.assert_array_equal(original_data.region_mask, loaded_data.region_mask)
        
        # Test features
        for key in original_data.features:
            np.testing.assert_array_equal(
                original_data.features[key], 
                loaded_data.features[key]
            )
        
        print("✅ Save/load functionality working correctly")
        
        # Clean up
        Path(test_file).unlink()
    
    def test_metadata_completeness(self):
        """Test metadata generation and completeness."""
        print("\n📊 Testing Metadata Generation...")
        
        # Generate with custom parameters
        custom_params = {
            'hotspot_centers': [(300, 300), (700, 700)],
            'intensities': [0.9, 0.7],
            'sigma': 75
        }
        
        heatmap_data = self.generator.generate_heatmap(
            PatternType.GAUSSIAN_HOTSPOT,
            **custom_params
        )
        
        metadata = heatmap_data.metadata
        
        # Check required fields
        assert 'pattern_type' in metadata
        assert 'resolution_config' in metadata
        assert 'generation_params' in metadata
        assert 'scale_factor' in metadata
        
        # Check values
        assert metadata['pattern_type'] == 'gaussian_hotspot'
        assert metadata['scale_factor'] == (20, 20)
        assert metadata['generation_params'] == custom_params
        
        print("✅ Metadata contains all required information")
    
    def run_all_tests(self):
        """Run all test methods."""
        print("Starting comprehensive test suite...")
        
        # Run individual tests
        self.test_initialization()
        self.test_coordinate_mapping()
        self.test_gaussian_hotspot_generation()
        self.test_horizontal_gradient_generation()
        self.test_vertical_gradient_generation()
        self.test_radial_gradient_generation()
        self.test_downsampling_accuracy()
        self.test_feature_extraction()
        self.test_save_load_functionality()
        self.test_metadata_completeness()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        
        return True
    
    def generate_test_visualization(self):
        """Generate comprehensive test visualization."""
        print("\n🎨 Generating Test Visualization...")
        
        # Generate multiple test patterns for visualization
        patterns_to_test = [
            ("Gaussian Hotspot", PatternType.GAUSSIAN_HOTSPOT, {
                'hotspot_centers': [(250, 750), (750, 250)],
                'intensities': [1.0, 0.6],
                'sigma': 80
            }),
            ("Horizontal Gradient", PatternType.HORIZONTAL_GRADIENT, {
                'direction': 'left_to_right',
                'min_value': 0.2,
                'max_value': 0.8
            }),
            ("Vertical Gradient", PatternType.VERTICAL_GRADIENT, {
                'direction': 'top_to_bottom',
                'min_value': 0.1,
                'max_value': 0.9
            }),
            ("H-Gradient Reverse", PatternType.HORIZONTAL_GRADIENT, {
                'direction': 'right_to_left',
                'min_value': 0.3,
                'max_value': 0.7
            }),
            ("V-Gradient Reverse", PatternType.VERTICAL_GRADIENT, {
                'direction': 'bottom_to_top',
                'min_value': 0.25,
                'max_value': 0.75
            }),
            ("Multiple Hotspots", PatternType.GAUSSIAN_HOTSPOT, {
                'hotspot_centers': [(200, 200), (500, 500), (800, 800)],
                'intensities': [0.8, 1.0, 0.7],
                'sigma': 60
            }),
            ("Radial Center→Out", PatternType.RADIAL_GRADIENT, {
                'direction': 'center_outward',
                'center': (500, 500),
                'min_value': 0.1,
                'max_value': 0.9,
                'max_radius': 400
            }),
            ("Radial Out→Center", PatternType.RADIAL_GRADIENT, {
                'direction': 'outward_center',
                'center': (400, 600),
                'min_value': 0.2,
                'max_value': 0.8,
                'max_radius': 350
            })
        ]
        
        # Create larger visualization grid
        fig, axes = plt.subplots(8, 3, figsize=(15, 32))
        fig.suptitle("HeatmapGenerator Test Results - All Pattern Types", fontsize=20)
        
        for i, (name, pattern_type, params) in enumerate(patterns_to_test):
            heatmap_data = self.generator.generate_heatmap(pattern_type, **params)
            
            # Display layer
            im1 = axes[i, 0].imshow(heatmap_data.display_layer, cmap='hot', aspect='equal')
            axes[i, 0].set_title(f'{name} - Display Layer', fontsize=12)
            axes[i, 0].set_xlabel('X Coordinate')
            axes[i, 0].set_ylabel('Y Coordinate')
            
            # Decision grid
            im2 = axes[i, 1].imshow(heatmap_data.decision_grid, cmap='hot', aspect='equal')
            axes[i, 1].set_title(f'{name} - Decision Grid', fontsize=12)
            axes[i, 1].set_xlabel('Grid X')
            axes[i, 1].set_ylabel('Grid Y')
            
            # Buildings feature
            axes[i, 2].imshow(heatmap_data.features['buildings'], cmap='gray', aspect='equal')
            axes[i, 2].set_title(f'{name} - Buildings', fontsize=12)
            
            # Add colorbars for better visualization
            if i == 0:
                plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
                plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
        
        plt.tight_layout()
        
        # Save visualization
        output_path = "test_results_comprehensive.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Test visualization saved to: {output_path}")
        
        # Show plot
        plt.show()
        
        return output_path


def run_performance_test():
    """Run performance benchmarking."""
    print("\n⚡ Performance Testing...")
    
    generator = HeatmapGenerator()
    
    import time
    
    # Time generation
    start_time = time.time()
    heatmap_data = generator.generate_heatmap(PatternType.GAUSSIAN_HOTSPOT)
    generation_time = time.time() - start_time
    
    print(f"✅ Generation time: {generation_time:.3f}s")
    
    # Memory usage estimation
    memory_usage = (
        heatmap_data.display_layer.nbytes +
        heatmap_data.decision_grid.nbytes +
        sum(arr.nbytes for arr in heatmap_data.features.values()) +
        heatmap_data.region_mask.nbytes
    ) / (1024 * 1024)  # Convert to MB
    
    print(f"✅ Memory usage: {memory_usage:.1f}MB")
    
    return generation_time, memory_usage


if __name__ == "__main__":
    # Run comprehensive test suite
    tester = TestHeatmapGenerator()
    
    try:
        # Run all tests
        tester.run_all_tests()
        
        # Generate visualization
        tester.generate_test_visualization()
        
        # Performance testing
        generation_time, memory_usage = run_performance_test()
        
        print(f"\n📈 Performance Summary:")
        print(f"   Generation: {generation_time:.3f}s")
        print(f"   Memory: {memory_usage:.1f}MB")
        print(f"   Status: {'✅ EXCELLENT' if generation_time < 0.1 else '⚠️ ACCEPTABLE'}")
        
        print(f"\n🚀 HeatmapGenerator is ready for production use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)