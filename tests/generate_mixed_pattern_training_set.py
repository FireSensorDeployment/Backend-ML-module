"""
Generate training set with MIXED patterns - multiple fire risk types per map
- Combine Gaussian + Gradient + Radial in single scenarios
- Limited gradient coverage
- Show both fire risk AND buildings in visualization
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

class MixedPatternGenerator:
    """Generator for mixed fire risk patterns"""
    
    def __init__(self):
        self.generator = HeatmapGenerator()
    
    def generate_mixed_fire_risk(self):
        """Generate fire risk map with multiple overlapping patterns"""
        
        height, width = self.generator.config.display_size
        combined_heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Randomly decide which patterns to include (1-3 patterns per map)
        num_patterns = np.random.randint(1, 4)  # 2-3 patterns per map
        
        available_patterns = [
            'gaussian_hotspot',
            'limited_horizontal_gradient', 
            'limited_vertical_gradient',
            'radial_gradient'
        ]
        
        selected_patterns = np.random.choice(available_patterns, size=num_patterns, replace=False)
        pattern_info = []
        
        for pattern_name in selected_patterns:
            if pattern_name == 'gaussian_hotspot':
                # Add 1-3 gaussian hotspots
                layer = self._generate_limited_gaussian()
                pattern_info.append({'type': 'gaussian', 'params': 'multiple_hotspots'})
                
            elif pattern_name == 'limited_horizontal_gradient':
                # Add limited horizontal gradient
                layer = self._generate_constrained_horizontal_gradient()
                pattern_info.append({'type': 'horizontal_gradient', 'params': 'limited'})
                
            elif pattern_name == 'limited_vertical_gradient':
                # Add limited vertical gradient  
                layer = self._generate_constrained_vertical_gradient()
                pattern_info.append({'type': 'vertical_gradient', 'params': 'limited'})
                
            elif pattern_name == 'radial_gradient':
                # Add localized radial gradient
                layer = self._generate_localized_radial()
                pattern_info.append({'type': 'radial_gradient', 'params': 'localized'})
            
            # Combine with existing (take maximum to avoid overriding)
            combined_heatmap = np.maximum(combined_heatmap, layer)
        
        # Downsample to decision grid
        decision_grid = self.generator._downsample_to_grid(combined_heatmap)
        
        # Create metadata
        metadata = {
            'pattern_type': 'mixed',
            'included_patterns': selected_patterns.tolist(),
            'pattern_details': pattern_info,
            'generation_params': {'mixed': True}
        }
        
        # Create HeatmapData-like structure
        from heatmap_generator import HeatmapData
        return HeatmapData(
            display_layer=combined_heatmap,
            decision_grid=decision_grid, 
            metadata=metadata
        )
    
    def _generate_limited_gaussian(self):
        """Generate 1-5 gaussian hotspots with varied sizes - more diversity"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # 1-5 hotspots with different size categories
        num_hotspots = np.random.randint(1, 6)  # Increased from 1-3 to 1-5
        
        for i in range(num_hotspots):
            # Random position (avoid edges)
            center_x = np.random.randint(80, width - 80)
            center_y = np.random.randint(80, height - 80)
            
            # Three size categories for more variety
            size_category = np.random.choice(['small', 'medium', 'large'], 
                                           p=[0.4, 0.4, 0.2])  # Bias toward smaller hotspots
            
            if size_category == 'small':
                # Small concentrated hotspots
                intensity = np.random.uniform(0.5, 0.9)  # Higher intensity
                sigma = np.random.uniform(15, 40)  # Small spread
            elif size_category == 'medium':
                # Medium hotspots  
                intensity = np.random.uniform(0.4, 0.7)  # Medium intensity
                sigma = np.random.uniform(40, 80)  # Medium spread
            else:  # large
                # Large diffuse hotspots
                intensity = np.random.uniform(0.3, 0.6)  # Lower intensity but wider
                sigma = np.random.uniform(80, 150)  # Large spread
            
            # Generate single hotspot
            y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
            distance_sq = (x_coords - center_x)**2 + (y_coords - center_y)**2
            gaussian = intensity * np.exp(-distance_sq / (2 * sigma**2))
            
            heatmap = np.maximum(heatmap, gaussian)  # Use max to combine
        
        return heatmap
    
    def _generate_constrained_horizontal_gradient(self):
        """Generate horizontal gradient from edges - simulating high-risk border zones"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Define gradient region - MAX 1/4 of total width
        max_region_width = width // 4  # 1/4 of total width
        region_width = np.random.randint(width // 8, max_region_width)  # Between 1/8 and 1/4
        
        # Choose which edge: left (0) or right (1)
        from_right_edge = np.random.choice([True, False])
        
        if from_right_edge:
            # Start from right edge, gradient goes inward (right to left)
            region_start = width - region_width
            region_end = width
            # High risk at edge (right), fading inward (left)
            max_val = np.random.uniform(0.5, 0.8)  # Higher intensity at edge
            min_val = np.random.uniform(0.1, 0.2)  # Fade to low
            gradient_1d = np.linspace(min_val, max_val, region_width)  # min -> max (left to right)
        else:
            # Start from left edge, gradient goes inward (left to right) 
            region_start = 0
            region_end = region_width
            # High risk at edge (left), fading inward (right)
            max_val = np.random.uniform(0.5, 0.8)  # Higher intensity at edge
            min_val = np.random.uniform(0.1, 0.2)  # Fade to low
            gradient_1d = np.linspace(max_val, min_val, region_width)  # max -> min (left to right)
        
        # Apply to vertical strip
        heatmap[:, region_start:region_end] = gradient_1d
        
        return heatmap
    
    def _generate_constrained_vertical_gradient(self):
        """Generate vertical gradient from edges - simulating high-risk border zones"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Define gradient region - MAX 1/4 of total height
        max_region_height = height // 4  # 1/4 of total height
        region_height = np.random.randint(height // 8, max_region_height)  # Between 1/8 and 1/4
        
        # Choose which edge: top (0) or bottom (1)
        from_bottom_edge = np.random.choice([True, False])
        
        if from_bottom_edge:
            # Start from bottom edge, gradient goes inward (bottom to top)
            region_start = height - region_height
            region_end = height
            # High risk at edge (bottom), fading inward (top)
            max_val = np.random.uniform(0.5, 0.8)  # Higher intensity at edge
            min_val = np.random.uniform(0.1, 0.2)  # Fade to low
            gradient_1d = np.linspace(min_val, max_val, region_height)  # min -> max (top to bottom)
        else:
            # Start from top edge, gradient goes inward (top to bottom)
            region_start = 0
            region_end = region_height
            # High risk at edge (top), fading inward (bottom)
            max_val = np.random.uniform(0.5, 0.8)  # Higher intensity at edge
            min_val = np.random.uniform(0.1, 0.2)  # Fade to low
            gradient_1d = np.linspace(max_val, min_val, region_height)  # max -> min (top to bottom)
        
        # Apply to horizontal strip
        heatmap[region_start:region_end, :] = gradient_1d.reshape(-1, 1)
        
        return heatmap
    
    def _generate_localized_radial(self):
        """Generate radial gradient with limited radius"""
        height, width = self.generator.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Random center
        center_x = np.random.randint(width // 4, 3 * width // 4)
        center_y = np.random.randint(height // 4, 3 * height // 4)
        
        # Limited radius and intensity
        max_radius = np.random.uniform(80, 200)  # Smaller radius
        min_val = np.random.uniform(0.0, 0.1)
        max_val = np.random.uniform(0.3, 0.6)  # Lower max
        
        # Create coordinates
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Apply gradient only within radius
        mask = distance <= max_radius
        normalized_dist = np.clip(distance / max_radius, 0, 1)
        
        # Outward gradient (center is high risk)
        if np.random.choice([True, False]):
            gradient_vals = max_val - normalized_dist * (max_val - min_val)
        else:
            gradient_vals = min_val + normalized_dist * (max_val - min_val)
        
        heatmap[mask] = gradient_vals[mask]
        
        return heatmap

def generate_mixed_training_set():
    """Generate 50 scenarios with mixed patterns"""
    
    mixed_gen = MixedPatternGenerator()
    building_gen = BuildingMapGenerator()
    
    print("=== Generating Mixed Pattern Training Set (50 scenarios) ===")
    
    training_scenarios = []
    
    for i in range(50):
        print(f"Generating scenario {i+1}/50...")
        
        # Generate mixed fire risk patterns
        fire_data = mixed_gen.generate_mixed_fire_risk()
        
        # Random building pattern
        building_pattern = np.random.choice([
            BuildingPattern.SPARSE,
            BuildingPattern.RANDOM, 
            BuildingPattern.CLUSTERED
        ])
        
        building_data = building_gen.generate_buildings(building_pattern)
        
        # Create scenario
        scenario = {
            'fire_risk': fire_data.decision_grid,
            'buildings': building_data.decision_grid,
            'metadata': {
                'scenario_id': i+1,
                'fire_patterns': fire_data.metadata['included_patterns'],
                'building_pattern': building_pattern.value,
                'fire_params': fire_data.metadata,
                'building_params': building_data.metadata['generation_params']
            }
        }
        
        training_scenarios.append(scenario)
        
        # Stats
        high_risk_cells = np.sum(fire_data.decision_grid > 0.7)
        building_cells = np.sum(building_data.decision_grid > 0.1)
        total_cells = fire_data.decision_grid.size
        
        patterns_str = '+'.join([p[:4] for p in fire_data.metadata['included_patterns']])
        print(f"    Patterns: {patterns_str}, High risk: {high_risk_cells/total_cells:.1%}, Buildings: {building_cells}")
    
    # Save training set
    output_file = 'mixed_pattern_training_set_50.npz'
    np.savez_compressed(output_file, 
                       scenarios=training_scenarios,
                       batch_size=len(training_scenarios))
    
    print(f"\n✅ Mixed pattern training set saved to {output_file}")
    
    return training_scenarios

def visualize_mixed_scenarios(scenarios):
    """Visualize scenarios using heatmap coloring like helper_functions_demo"""
    
    print("\n=== Creating Mixed Pattern Visualization (Heatmap Style) ===")
    
    # Create large figure: 10x5 grid for 50 scenarios  
    fig, axes = plt.subplots(10, 5, figsize=(25, 50))
    fig.suptitle('Mixed Pattern Training Set: Fire Risk Heatmap + Buildings (50 scenarios)', fontsize=18, y=0.995)
    
    for i, scenario in enumerate(scenarios):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Create combined visualization: fire risk heatmap + building overlay
        # Start with fire risk as base heatmap
        combined_data = scenario['fire_risk'].copy()
        
        # Overlay buildings: building intensity determines brightness
        # Buildings are shown as white/bright pixels on top of fire heatmap
        building_mask = scenario['buildings'] > 0.1
        building_intensity = scenario['buildings'][building_mask]
        
        # For building cells, use white color scaled by building density
        # Higher building density = brighter white
        combined_data[building_mask] = np.maximum(
            combined_data[building_mask], 
            0.7 + building_intensity * 0.3  # Scale to 0.7-1.0 range (bright white)
        )
        
        # Show combined heatmap (fire + buildings)
        fire_heatmap = ax.imshow(combined_data, cmap='hot', vmin=0, vmax=1, alpha=1.0)
        
        # Title with pattern info
        patterns = scenario['metadata']['fire_patterns']
        building_pattern = scenario['metadata']['building_pattern']
        
        # Stats
        high_risk = np.sum(scenario['fire_risk'] > 0.7)
        total_cells = scenario['fire_risk'].size
        building_cells = np.sum(scenario['buildings'] > 0.1)
        
        patterns_short = '+'.join([p[:4] for p in patterns])
        title = f"#{i+1}: {patterns_short}\n{building_pattern[:4]}, HR:{high_risk/total_cells:.1%}, B:{building_cells}"
        ax.set_title(title, fontsize=10, color='white')  # White text for visibility on dark background
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Add legend (no colorbar needed)
    legend_text = "🔥 Fire Risk (Black→Red→Orange→Yellow)  ⬜ Buildings (White intensity = density)"
    fig.text(0.5, 0.002, legend_text, ha='center', fontsize=14, color='white',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.8))
    
    # Set dark background
    fig.patch.set_facecolor('black')
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = 'mixed_pattern_heatmap_visualization.png'
    plt.savefig(viz_file, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"✅ Mixed pattern heatmap visualization saved to {viz_file}")
    
    return viz_file

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(789)  # New seed for updated generator
    
    # Generate mixed training set
    scenarios = generate_mixed_training_set()
    
    # Create visualization
    viz_file = visualize_mixed_scenarios(scenarios)
    
    print(f"\n🎯 Complete! Generated:")
    print(f"   📊 mixed_pattern_training_set_50.npz - 50 mixed-pattern scenarios") 
    print(f"   🖼️  {viz_file} - Fire + Buildings visualization")
    print(f"\n🚀 Ready for RL training with realistic mixed scenarios!")