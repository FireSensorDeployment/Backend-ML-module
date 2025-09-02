"""
Visualize fire patterns at display resolution (upscaled) from existing npz file
- Show high-resolution fire patterns as they appear in display layer
- Compare with low-resolution decision grid
"""

import sys
import os
current_dir = os.path.dirname(__file__)
src_data_path = os.path.join(current_dir, '..', 'src', 'data')
sys.path.insert(0, src_data_path)

from heatmap_generator import HeatmapGenerator, PatternType
from building_generator import BuildingMapGenerator, BuildingPattern
import numpy as np
import matplotlib.pyplot as plt

def load_and_visualize_display_layer():
    """Load npz file and visualize both decision grid and upscaled display layer"""
    
    # Load the mixed pattern data
    data = np.load('mixed_pattern_training_set_50.npz', allow_pickle=True)
    scenarios = data['scenarios']
    
    print(f"=== Loaded {len(scenarios)} scenarios from npz file ===")
    
    # We need to regenerate the display layers since npz only contains decision grids
    # Let's recreate a few scenarios to show display vs decision comparison
    
    mixed_gen = MixedPatternGenerator()
    building_gen = BuildingMapGenerator()
    
    # Select first 9 scenarios for visualization 
    num_scenarios = min(9, len(scenarios))
    display_scenarios = []
    
    print(f"Re-generating display layers for {num_scenarios} scenarios...")
    
    for i in range(num_scenarios):
        print(f"Processing scenario {i+1}/{num_scenarios}...")
        
        # Get pattern types from metadata 
        original_scenario = scenarios[i]
        patterns = original_scenario['metadata']['fire_patterns']
        building_pattern_name = original_scenario['metadata']['building_pattern']
        
        # Regenerate with same patterns (results will be different due to randomness)
        np.random.seed(42 + i)  # Different seed per scenario
        fire_data = mixed_gen.generate_mixed_fire_risk()
        
        # Match building pattern
        building_pattern = BuildingPattern.SPARSE  # Default
        if 'random' in building_pattern_name.lower():
            building_pattern = BuildingPattern.RANDOM
        elif 'cluster' in building_pattern_name.lower():
            building_pattern = BuildingPattern.CLUSTERED
            
        building_data = building_gen.generate_buildings(building_pattern)
        
        display_scenarios.append({
            'display_fire': fire_data.display_layer,
            'decision_fire': fire_data.decision_grid,
            'display_buildings': building_data.display_layer,
            'decision_buildings': building_data.decision_grid,
            'patterns': patterns,
            'scenario_id': i+1
        })
    
    # Create comparison visualization
    create_display_comparison(display_scenarios)
    
    return display_scenarios

class MixedPatternGenerator:
    """Simplified version to regenerate display layers"""
    
    def __init__(self):
        from heatmap_generator import HeatmapGenerator
        self.generator = HeatmapGenerator()
    
    def generate_mixed_fire_risk(self):
        """Generate fire risk with display layer"""
        height, width = self.generator.config.display_size
        combined_heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Randomly select patterns
        available_patterns = [
            'gaussian_hotspot',
            'limited_horizontal_gradient', 
            'limited_vertical_gradient',
            'radial_gradient'
        ]
        
        num_patterns = np.random.randint(2, 4)
        selected_patterns = np.random.choice(available_patterns, size=num_patterns, replace=False)
        
        for pattern_name in selected_patterns:
            if pattern_name == 'gaussian_hotspot':
                layer = self._generate_limited_gaussian()
            elif pattern_name == 'limited_horizontal_gradient':
                layer = self._generate_constrained_horizontal_gradient()
            elif pattern_name == 'limited_vertical_gradient':
                layer = self._generate_constrained_vertical_gradient()
            elif pattern_name == 'radial_gradient':
                layer = self._generate_localized_radial()
            
            combined_heatmap = np.maximum(combined_heatmap, layer)
        
        # Downsample to decision grid
        decision_grid = self.generator._downsample_to_grid(combined_heatmap)
        
        # Create HeatmapData-like structure
        from heatmap_generator import HeatmapData
        return HeatmapData(
            display_layer=combined_heatmap,
            decision_grid=decision_grid,
            metadata={'pattern_type': 'mixed', 'included_patterns': selected_patterns.tolist()}
        )
    
    def _generate_limited_gaussian(self):
        """Generate gaussian hotspots with variety"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        num_hotspots = np.random.randint(1, 4)
        
        for _ in range(num_hotspots):
            center_x = np.random.randint(80, width - 80)
            center_y = np.random.randint(80, height - 80)
            
            size_category = np.random.choice(['small', 'medium', 'large'], p=[0.4, 0.4, 0.2])
            
            if size_category == 'small':
                intensity = np.random.uniform(0.5, 0.9)
                sigma = np.random.uniform(15, 40)
            elif size_category == 'medium':
                intensity = np.random.uniform(0.4, 0.7)
                sigma = np.random.uniform(40, 80)
            else:
                intensity = np.random.uniform(0.3, 0.6)
                sigma = np.random.uniform(80, 150)
            
            y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            distance_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
            gaussian = intensity * np.exp(-distance_sq / (2 * sigma**2))
            
            heatmap = np.maximum(heatmap, gaussian)
        
        return heatmap
    
    def _generate_constrained_horizontal_gradient(self):
        """Generate horizontal gradient from edges"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        max_region_width = width // 4
        region_width = np.random.randint(width // 8, max_region_width)
        from_right_edge = np.random.choice([True, False])
        
        if from_right_edge:
            region_start = width - region_width
            region_end = width
            max_val = np.random.uniform(0.5, 0.8)
            min_val = np.random.uniform(0.1, 0.2)
            gradient_1d = np.linspace(min_val, max_val, region_width)
        else:
            region_start = 0
            region_end = region_width
            max_val = np.random.uniform(0.5, 0.8)
            min_val = np.random.uniform(0.1, 0.2)
            gradient_1d = np.linspace(max_val, min_val, region_width)
        
        heatmap[:, region_start:region_end] = gradient_1d
        return heatmap
    
    def _generate_constrained_vertical_gradient(self):
        """Generate vertical gradient from edges"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        max_region_height = height // 4
        region_height = np.random.randint(height // 8, max_region_height)
        from_bottom_edge = np.random.choice([True, False])
        
        if from_bottom_edge:
            region_start = height - region_height
            region_end = height
            max_val = np.random.uniform(0.5, 0.8)
            min_val = np.random.uniform(0.1, 0.2)
            gradient_1d = np.linspace(min_val, max_val, region_height)
        else:
            region_start = 0
            region_end = region_height
            max_val = np.random.uniform(0.5, 0.8)
            min_val = np.random.uniform(0.1, 0.2)
            gradient_1d = np.linspace(max_val, min_val, region_height)
        
        heatmap[region_start:region_end, :] = gradient_1d.reshape(-1, 1)
        return heatmap
    
    def _generate_localized_radial(self):
        """Generate radial gradient"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        center_x = np.random.randint(width // 4, 3 * width // 4)
        center_y = np.random.randint(height // 4, 3 * height // 4)
        
        max_radius = np.random.uniform(80, 200)
        min_val = np.random.uniform(0.0, 0.1)
        max_val = np.random.uniform(0.3, 0.6)
        
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        mask = distance <= max_radius
        normalized_dist = np.clip(distance / max_radius, 0, 1)
        
        if np.random.choice([True, False]):
            gradient_vals = max_val - normalized_dist * (max_val - min_val)
        else:
            gradient_vals = min_val + normalized_dist * (max_val - min_val)
        
        heatmap[mask] = gradient_vals[mask]
        return heatmap

def create_display_comparison(scenarios):
    """Create visualization comparing decision grid vs display layer"""
    
    num_scenarios = len(scenarios)
    
    # Create large figure: 3 rows x 3 columns for 9 scenarios
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Mixed Pattern: High-Resolution Display Layer Visualization', fontsize=16)
    
    for i, scenario in enumerate(scenarios):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Create RGB image for display layer: Red=Fire Risk, White=Buildings, Dark=Empty
        rgb_image = np.zeros((*scenario['display_fire'].shape, 3))
        
        # Red channel: Fire risk
        rgb_image[:,:,0] = scenario['display_fire']
        
        # White for buildings
        building_intensity = scenario['display_buildings'] * 0.9
        rgb_image[:,:,0] = np.maximum(rgb_image[:,:,0], building_intensity)
        rgb_image[:,:,1] = building_intensity  
        rgb_image[:,:,2] = building_intensity
        
        # Dark background
        rgb_image[:,:,2] = np.maximum(rgb_image[:,:,2], 0.05)
        
        # Show combined image
        ax.imshow(rgb_image, vmin=0, vmax=1)
        
        # Title with pattern info
        patterns = scenario['patterns']
        patterns_short = '+'.join([p[:4] for p in patterns])
        
        # Stats for display layer
        high_risk_pixels = np.sum(scenario['display_fire'] > 0.7)
        total_pixels = scenario['display_fire'].size
        building_pixels = np.sum(scenario['display_buildings'] > 0.1)
        
        title = f"#{scenario['scenario_id']}: {patterns_short}\nHR:{high_risk_pixels/total_pixels:.1%}, Buildings:{building_pixels}"
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add legend
    legend_text = "🔴 Fire Risk (High-Res)  ⚪ Buildings  ⚫ Empty"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = 'display_layer_visualization.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight')
    print(f"✅ Display layer visualization saved to {viz_file}")
    
    return viz_file

if __name__ == "__main__":
    # Set random seed
    np.random.seed(555)
    
    # Load and visualize
    scenarios = load_and_visualize_display_layer()
    
    print(f"\n🎯 Complete! Generated high-resolution display layer visualization")
    print(f"   🖼️  display_layer_visualization.png - High-res fire patterns")
    print(f"\n✨ Display resolution shows smooth gradients and detailed fire patterns!")