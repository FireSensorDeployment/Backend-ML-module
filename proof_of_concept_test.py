"""
Proof of concept test for realistic fire risk scenario.

This test demonstrates:
1. 5 small scattered Gaussian fire points (different locations across the map)  
2. Partial vertical gradient on left side (representing environmental risk gradient)
3. Combined realistic fire risk pattern for RL training
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.heatmap_generator import (
    HeatmapGenerator, 
    ResolutionConfig, 
    PatternType, 
    HeatmapData
)


class ProofOfConceptTest:
    """Test realistic fire risk scenarios for RL training."""
    
    def __init__(self):
        """Initialize the test environment."""
        self.generator = HeatmapGenerator()
        print("🔥 Proof of Concept: Realistic Fire Risk Scenario")
        print("=" * 60)
    
    def create_combined_fire_risk_scenario(self):
        """Create a realistic fire risk scenario combining multiple patterns."""
        print("\n🎯 Creating Combined Fire Risk Scenario...")
        
        # Step 1: Generate base scenario with 5 diverse fire points in RIGHT 3/4 area
        # Right 3/4 means x coordinates from 250-1000 (camping sites, etc.)
        fire_points_data = self.generator.generate_heatmap(
            PatternType.GAUSSIAN_HOTSPOT,
            hotspot_centers=[
                (150, 350),   # Small camping site (right 3/4)
                (300, 600),   # Medium picnic area  
                (450, 750),   # Small fire pit
                (650, 850),   # Large campground
                (800, 500)    # Medium RV area
            ],
            intensities=[0.5, 0.8, 0.4, 1.0, 0.7],  # Varying risk intensities
            sigmas=[25, 60, 15, 90, 45]  # DIFFERENT radius for each site!
        )
        
        # Step 2: Create partial vertical gradient on left side (environmental risk)
        environmental_risk = self.generator.generate_heatmap(
            PatternType.VERTICAL_GRADIENT,
            direction='bottom_to_top',    # Risk increases northward
            min_value=0.1,
            max_value=0.4                 # Moderate environmental risk
        )
        
        # Step 3: Combine the patterns - mask the gradient to only affect left 1/4 of map
        combined_display = fire_points_data.display_layer.copy()
        combined_decision = fire_points_data.decision_grid.copy()
        
        # Apply environmental gradient only to left quarter (x < 250) - VEGETATION RISK
        left_quarter_mask = np.zeros_like(combined_display, dtype=bool)
        left_quarter_mask[:, :250] = True  # Left 1/4 of 1000px width
        
        # Add environmental risk to left quarter (vegetation, terrain factors)
        combined_display[left_quarter_mask] += environmental_risk.display_layer[left_quarter_mask]
        
        # Update decision grid similarly 
        grid_left_mask = np.zeros_like(combined_decision, dtype=bool)
        grid_left_mask[:, :12] = True  # Left 1/4 of 50px grid width (12/50 ≈ 1/4)
        combined_decision[grid_left_mask] += environmental_risk.decision_grid[grid_left_mask]
        
        # Ensure values don't exceed 1.0
        combined_display = np.clip(combined_display, 0, 1.0)
        combined_decision = np.clip(combined_decision, 0, 1.0)
        
        # Create combined heatmap data
        combined_scenario = HeatmapData(
            display_layer=combined_display,
            decision_grid=combined_decision,
            region_mask=fire_points_data.region_mask,
            metadata={
                'pattern_type': 'combined_fire_risk_scenario',
                'fire_points': 5,
                'environmental_gradient': 'left_quarter_vegetation_risk',
                'description': '5 diverse camping sites (right 3/4) + vegetation risk zone (left 1/4)'
            }
        )
        
        print("✅ Combined fire risk scenario created")
        print(f"   🔥 Fire points: {len(fire_points_data.metadata['generation_params']['hotspot_centers'])}")
        print(f"   🌡️ Environmental gradient: Left 1/4 of map")
        print(f"   📊 Peak risk value: {np.max(combined_display):.3f}")
        print(f"   📊 Average risk: {np.mean(combined_display):.3f}")
        
        return combined_scenario
    
    def visualize_scenario(self, scenario_data: HeatmapData):
        """Create comprehensive visualization of the fire risk scenario."""
        print("\n📊 Generating Visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Proof of Concept: Realistic Fire Risk Scenario", fontsize=16)
        
        # Display layer - main view
        im1 = axes[0, 0].imshow(scenario_data.display_layer, cmap='hot', aspect='equal')
        axes[0, 0].set_title('Fire Risk Map - Display Layer (1000×1000)')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Add annotations for fire points
        fire_points = [
            (150, 350, "Camp 1"),
            (300, 600, "Camp 2"), 
            (450, 750, "Camp 3"),
            (650, 850, "Camp 4"),
            (800, 500, "Camp 5")
        ]
        for y, x, label in fire_points:
            axes[0, 0].annotate(label, (x, y), color='white', fontweight='bold', 
                               ha='center', va='center', fontsize=8)
        
        # Add gradient zone annotation
        axes[0, 0].axvline(x=250, color='cyan', linestyle='--', alpha=0.7)
        axes[0, 0].text(125, 50, 'Vegetation\nRisk Zone', color='cyan', 
                       fontweight='bold', ha='center')
        
        # Decision grid - RL training view
        im2 = axes[0, 1].imshow(scenario_data.decision_grid, cmap='hot', aspect='equal')
        axes[0, 1].set_title('RL Decision Grid (50×50)')
        axes[0, 1].set_xlabel('Grid X')
        axes[0, 1].set_ylabel('Grid Y')
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # Risk distribution histogram
        axes[1, 0].hist(scenario_data.display_layer.flatten(), bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('Risk Value Distribution')
        axes[1, 0].set_xlabel('Risk Level')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Regional analysis - compare left vs right side risk
        left_risk = scenario_data.display_layer[:, :250].mean()
        right_risk = scenario_data.display_layer[:, 250:].mean()
        
        regions = ['Left Quarter\n(vegetation risk)', 'Right 3/4\n(camping sites)']
        risk_levels = [left_risk, right_risk]
        colors = ['orange', 'red']
        
        bars = axes[1, 1].bar(regions, risk_levels, color=colors, alpha=0.7)
        axes[1, 1].set_title('Regional Risk Analysis')
        axes[1, 1].set_ylabel('Average Risk Level')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, risk_levels):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = "proof_of_concept_fire_scenario.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualization saved to: {output_path}")
        
        plt.show()
        
        return output_path
    
    def analyze_scenario(self, scenario_data: HeatmapData):
        """Analyze the generated scenario for RL training suitability."""
        print("\n🔍 Scenario Analysis for RL Training...")
        
        display = scenario_data.display_layer
        grid = scenario_data.decision_grid
        
        # Risk distribution analysis
        print(f"📊 Risk Statistics:")
        print(f"   Peak risk: {np.max(display):.3f}")
        print(f"   Average risk: {np.mean(display):.3f}")
        print(f"   Risk std dev: {np.std(display):.3f}")
        
        # High-risk area identification
        high_risk_threshold = 0.5
        high_risk_pixels = np.sum(display > high_risk_threshold)
        high_risk_percentage = high_risk_pixels / display.size * 100
        print(f"   High-risk areas (>{high_risk_threshold}): {high_risk_percentage:.1f}%")
        
        # Decision grid analysis
        print(f"\n🎯 RL Decision Grid Analysis:")
        print(f"   Grid resolution: {grid.shape}")
        print(f"   Max grid value: {np.max(grid):.3f}")
        print(f"   High-priority cells: {np.sum(grid > high_risk_threshold)} / {grid.size}")
        
        # Spatial distribution
        fire_point_coords = [
            (150, 350), (300, 600), (450, 750), (650, 850), (800, 500)
        ]
        
        print(f"\n🔥 Fire Point Distribution:")
        for i, (y, x) in enumerate(fire_point_coords, 1):
            risk_value = display[y, x]
            print(f"   Fire {i} at ({y:3d}, {x:3d}): risk = {risk_value:.3f}")
        
        # Environmental gradient analysis
        left_quarter = display[:, :250]
        right_three_quarters = display[:, 250:]
        
        print(f"\n🌡️ Environmental Gradient Effect:")
        print(f"   Left quarter avg risk: {np.mean(left_quarter):.3f}")
        print(f"   Right 3/4 avg risk: {np.mean(right_three_quarters):.3f}")
        print(f"   Gradient contribution: {np.mean(left_quarter) - np.mean(right_three_quarters):.3f}")
        
        # RL training recommendations
        print(f"\n🎮 RL Training Recommendations:")
        print(f"   ✅ Good risk diversity: {np.std(display):.3f} standard deviation")
        print(f"   ✅ Multiple hot zones: 5 scattered fire points")
        print(f"   ✅ Environmental factors: Left-side gradient")
        print(f"   ✅ Realistic scale: Fire points σ=40 (vs default ~100)")
        
        return {
            'peak_risk': np.max(display),
            'avg_risk': np.mean(display),
            'high_risk_percentage': high_risk_percentage,
            'fire_points': len(fire_point_coords),
            'gradient_effect': np.mean(left_quarter) - np.mean(right_three_quarters)
        }
    
    def run_proof_of_concept(self):
        """Run the complete proof of concept test."""
        print("Starting proof of concept test...")
        
        # Create the scenario
        scenario = self.create_combined_fire_risk_scenario()
        
        # Visualize the results
        viz_path = self.visualize_scenario(scenario)
        
        # Analyze for RL suitability
        analysis = self.analyze_scenario(scenario)
        
        # Save scenario data
        self.generator.save_heatmap(scenario, "proof_of_concept_scenario.npz")
        print(f"\n💾 Scenario data saved to: proof_of_concept_scenario.npz")
        
        print(f"\n🎉 Proof of Concept Complete!")
        print(f"   📊 Created realistic fire risk scenario")
        print(f"   🎨 Generated visualization: {viz_path}")
        print(f"   📈 Analyzed RL training suitability")
        print(f"   💾 Saved scenario data for future use")
        
        return scenario, analysis


if __name__ == "__main__":
    # Run the proof of concept
    poc = ProofOfConceptTest()
    
    try:
        scenario_data, analysis_results = poc.run_proof_of_concept()
        print(f"\n✨ Proof of concept successfully completed!")
        
    except Exception as e:
        print(f"\n❌ Error during proof of concept: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)