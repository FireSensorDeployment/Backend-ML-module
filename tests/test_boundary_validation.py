"""
Comprehensive test for boundary validation functionality.
Tests coordinate mapping edge cases and safety mechanisms.
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.heatmap_generator import HeatmapGenerator, ResolutionConfig

def test_boundary_validation():
    """Test boundary validation with various edge cases."""
    print("🛡️ Testing Boundary Validation...")
    
    generator = HeatmapGenerator()
    print(f"   Grid size: {generator.config.grid_size}")
    print(f"   Display size: {generator.config.display_size}")
    print(f"   Scale factor: {generator.scale_factor}")
    
    # Test cases: (input, expected_output, description)
    grid_test_cases = [
        # Normal cases
        ((0, 0), (0, 0), "Top-left corner"),
        ((25, 25), (500, 500), "Center position"),
        ((49, 49), (980, 980), "Bottom-right corner"),
        
        # Edge cases - should be clamped
        ((-1, -1), (0, 0), "Negative coordinates -> clamped to (0,0)"),
        ((50, 50), (980, 980), "Exactly at boundary -> clamped to (49,49)"),
        ((100, 100), (980, 980), "Far out of bounds -> clamped to (49,49)"),
        ((-10, 25), (0, 500), "Mixed valid/invalid -> partially clamped"),
    ]
    
    print("\n🗺️ Testing Grid -> Display Coordinate Mapping:")
    for (grid_r, grid_c), expected, description in grid_test_cases:
        result = generator.grid_to_display_coords(grid_r, grid_c, safe=True)
        print(f"   {description}")
        print(f"      Input: grid({grid_r}, {grid_c}) -> Output: display{result}")
        print(f"      Expected: {expected}, Match: {'✅' if result == expected else '❌'}")
        assert result == expected, f"Failed for {description}"
    
    # Test display -> grid mapping
    display_test_cases = [
        # Normal cases  
        ((0, 0), (0, 0), "Top-left corner"),
        ((500, 500), (25, 25), "Center position"),
        ((980, 980), (49, 49), "Bottom-right valid"),
        
        # Edge cases
        ((-1, -1), (0, 0), "Negative -> clamped"),
        ((1000, 1000), (49, 49), "At boundary -> clamped"),
        ((1500, 1200), (49, 49), "Far out -> clamped"), 
        ((-5, 500), (0, 25), "Mixed -> partially clamped"),
    ]
    
    print("\n🖼️ Testing Display -> Grid Coordinate Mapping:")
    for (disp_r, disp_c), expected, description in display_test_cases:
        result = generator.display_to_grid_coords(disp_r, disp_c, safe=True)
        print(f"   {description}")  
        print(f"      Input: display({disp_r}, {disp_c}) -> Output: grid{result}")
        print(f"      Expected: {expected}, Match: {'✅' if result == expected else '❌'}")
        assert result == expected, f"Failed for {description}"
    
    # Test round-trip accuracy with clamping
    print("\n🔄 Testing Round-trip Accuracy with Boundary Clamping:")
    test_coords = [(-5, -5), (0, 0), (25, 25), (49, 49), (60, 60), (100, 100)]
    
    for grid_r, grid_c in test_coords:
        # Grid -> Display -> Grid (should be clamped consistently)
        display_coords = generator.grid_to_display_coords(grid_r, grid_c, safe=True)
        back_to_grid = generator.display_to_grid_coords(*display_coords, safe=True)
        
        # After clamping, coordinates should be valid
        clamped_r = max(0, min(grid_r, 49))
        clamped_c = max(0, min(grid_c, 49))
        expected = (clamped_r, clamped_c)
        
        print(f"   Input: grid({grid_r}, {grid_c})")
        print(f"   Clamped: {expected} -> Display: {display_coords} -> Grid: {back_to_grid}")
        print(f"   Consistent: {'✅' if back_to_grid == expected else '❌'}")
        assert back_to_grid == expected, f"Round-trip failed for ({grid_r}, {grid_c})"
    
    # Test unsafe mode (should work without clamping for valid coordinates)
    print("\n⚠️ Testing Unsafe Mode (no boundary checking):")
    valid_coord = generator.grid_to_display_coords(25, 25, safe=False)
    print(f"   Valid coordinate (25,25) in unsafe mode: {valid_coord} ✅")
    
    # Test validation functions directly
    print("\n🔍 Testing Validation Functions:")
    
    # Grid validation tests
    grid_validation_tests = [
        ((0, 0), True, "Top-left valid"),
        ((49, 49), True, "Bottom-right valid"), 
        ((-1, 0), False, "Negative row invalid"),
        ((50, 0), False, "Row at boundary invalid"),
        ((25, -1), False, "Negative col invalid"),
        ((25, 50), False, "Col at boundary invalid"),
    ]
    
    for coords, expected, desc in grid_validation_tests:
        result = generator._validate_grid_coords(*coords)
        print(f"   {desc}: {coords} -> {result} {'✅' if result == expected else '❌'}")
        assert result == expected, f"Grid validation failed for {desc}"
    
    # Display validation tests  
    display_validation_tests = [
        ((0, 0), True, "Top-left valid"),
        ((999, 999), True, "Bottom-right valid"),
        ((-1, 0), False, "Negative row invalid"),
        ((1000, 0), False, "Row at boundary invalid"), 
        ((500, -1), False, "Negative col invalid"),
        ((500, 1000), False, "Col at boundary invalid"),
    ]
    
    for coords, expected, desc in display_validation_tests:
        result = generator._validate_display_coords(*coords)
        print(f"   {desc}: {coords} -> {result} {'✅' if result == expected else '❌'}")
        assert result == expected, f"Display validation failed for {desc}"
    
    print("\n🎉 All Boundary Validation Tests Passed!")
    return True

if __name__ == "__main__":
    test_boundary_validation()