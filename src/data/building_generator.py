"""
Building distribution generator for wildfire sensor deployment scenarios.

This module provides BuildingMapGenerator class for creating synthetic building
distributions compatible with dual-resolution system for RL training environments.
Designed specifically for wildfire protection scenarios with sparse building patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class BuildingPattern(Enum):
    """Available building distribution patterns."""
    SPARSE = "sparse"           # Mountain/rural scattered buildings
    RANDOM = "random"           # Random distributed buildings  
    CLUSTERED = "clustered"     # Small town/resort clustered buildings


@dataclass
class BuildingData:
    """Container for generated building distribution data."""
    display_layer: np.ndarray      # High-resolution building mask (1000x1000)
    decision_grid: np.ndarray      # RL decision grid (50x50)
    metadata: Dict[str, Any]       # Generation parameters and statistics


class BuildingMapGenerator:
    """
    Building distribution generator for wildfire scenarios.
    
    Generates sparse building distributions typical of wildfire-prone areas
    where buildings are scattered throughout forested/mountainous regions.
    Maintains compatibility with HeatmapGenerator's dual-resolution system.
    """
    
    def __init__(self, resolution_config=None):
        """Initialize generator with resolution configuration."""
        try:
            from .heatmap_generator import ResolutionConfig
        except ImportError:
            from heatmap_generator import ResolutionConfig
        
        self.config = resolution_config or ResolutionConfig()
        self._validate_resolution_config()
    
    def _validate_resolution_config(self):
        """Ensure resolution configuration is mathematically compatible."""
        display_h, display_w = self.config.display_size
        grid_h, grid_w = self.config.grid_size
        
        if display_h % grid_h != 0 or display_w % grid_w != 0:
            raise ValueError(
                f"Display size {self.config.display_size} must be evenly divisible "
                f"by grid size {self.config.grid_size} for proper alignment"
            )
        
        self.scale_factor = (display_h // grid_h, display_w // grid_w)
    
    def generate_buildings(self, pattern_type: BuildingPattern = BuildingPattern.SPARSE, **kwargs) -> BuildingData:
        """
        Generate building distribution with specified pattern type.
        
        Args:
            pattern_type: Type of building distribution pattern
            **kwargs: Pattern-specific parameters
            
        Returns:
            BuildingData containing all generated layers
        """
        # Generate building mask at high resolution
        display_layer = self._generate_pattern(pattern_type, **kwargs)
        
        # Downsample to create decision grid
        decision_grid = self._downsample_to_grid(display_layer)
        
        # Calculate statistics for metadata
        total_pixels = display_layer.size
        building_pixels = np.sum(display_layer > 0)
        coverage_ratio = building_pixels / total_pixels
        
        metadata = {
            'pattern_type': pattern_type.value,
            'resolution_config': self.config,
            'generation_params': kwargs,
            'scale_factor': self.scale_factor,
            'building_pixels': int(building_pixels),
            'coverage_ratio': float(coverage_ratio),
            'total_clusters': self._count_clusters(display_layer)
        }
        
        return BuildingData(
            display_layer=display_layer,
            decision_grid=decision_grid,
            metadata=metadata
        )
    
    def _generate_pattern(self, pattern_type: BuildingPattern, **kwargs) -> np.ndarray:
        """Generate specific building pattern at display resolution."""
        if pattern_type == BuildingPattern.SPARSE:
            return self._generate_sparse_clusters(**kwargs)
        elif pattern_type == BuildingPattern.RANDOM:
            return self._generate_random_buildings(**kwargs)
        elif pattern_type == BuildingPattern.CLUSTERED:
            return self._generate_clustered_buildings(**kwargs)
        else:
            raise ValueError(f"Unknown building pattern: {pattern_type}")
    
    def _generate_sparse_clusters(self, **kwargs) -> np.ndarray:
        """
        Generate sparse building clusters typical of mountain/rural areas.
        
        Creates 3-5 small building clusters scattered across the map,
        representing isolated settlements in wildfire-prone regions.
        """
        height, width = self.config.display_size
        building_mask = np.zeros((height, width), dtype=np.float32)
        
        # Parameters with defaults
        cluster_count = kwargs.get('cluster_count', np.random.randint(3, 6))
        min_cluster_size = kwargs.get('min_cluster_size', 10)
        max_cluster_size = kwargs.get('max_cluster_size', 30)
        buildings_per_cluster = kwargs.get('buildings_per_cluster', (1, 3))
        edge_buffer = kwargs.get('edge_buffer', 5)
        
        clusters_created = 0
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if clusters_created >= cluster_count:
                break
                
            # Random position avoiding edges
            center_x = np.random.randint(edge_buffer, width - edge_buffer)
            center_y = np.random.randint(edge_buffer, height - edge_buffer)
            
            # Cluster dimensions
            cluster_w = np.random.randint(min_cluster_size, max_cluster_size + 1)
            cluster_h = np.random.randint(min_cluster_size, max_cluster_size + 1)
            
            # Check bounds
            if (center_x + cluster_w//2 >= width - edge_buffer or 
                center_y + cluster_h//2 >= height - edge_buffer):
                continue
            
            # Create cluster region
            x_start = max(0, center_x - cluster_w//2)
            x_end = min(width, center_x + cluster_w//2)
            y_start = max(0, center_y - cluster_h//2)  
            y_end = min(height, center_y + cluster_h//2)
            
            # Generate individual buildings within cluster
            num_buildings = np.random.randint(buildings_per_cluster[0], buildings_per_cluster[1] + 1)
            
            for _ in range(num_buildings):
                # Random building size (small structures)
                building_w = np.random.randint(3, 8)
                building_h = np.random.randint(3, 8)
                
                # Random position within cluster
                if x_end - x_start > building_w and y_end - y_start > building_h:
                    bx = np.random.randint(x_start, x_end - building_w)
                    by = np.random.randint(y_start, y_end - building_h)
                    
                    # Place building
                    building_mask[by:by+building_h, bx:bx+building_w] = 1.0
            
            clusters_created += 1
        
        return building_mask
    
    def _generate_random_buildings(self, **kwargs) -> np.ndarray:
        """Generate randomly distributed individual buildings."""
        height, width = self.config.display_size
        building_mask = np.zeros((height, width), dtype=np.float32)
        
        # Parameters
        building_count = kwargs.get('building_count', np.random.randint(5, 13))
        min_building_size = kwargs.get('min_building_size', 5)
        max_building_size = kwargs.get('max_building_size', 15)
        edge_buffer = kwargs.get('edge_buffer', 5)
        
        buildings_placed = 0
        max_attempts = building_count * 10
        
        for attempt in range(max_attempts):
            if buildings_placed >= building_count:
                break
                
            building_w = np.random.randint(min_building_size, max_building_size + 1)
            building_h = np.random.randint(min_building_size, max_building_size + 1)
            
            if width <= edge_buffer * 2 + building_w or height <= edge_buffer * 2 + building_h:
                continue
                
            x = np.random.randint(edge_buffer, width - edge_buffer - building_w)
            y = np.random.randint(edge_buffer, height - edge_buffer - building_h)
            
            region = building_mask[y:y+building_h, x:x+building_w]
            if np.any(region > 0):
                continue
            
            building_mask[y:y+building_h, x:x+building_w] = 1.0
            buildings_placed += 1
        
        return building_mask
    
    def _generate_clustered_buildings(self, **kwargs) -> np.ndarray:
        """Generate 1-2 large building clusters (towns/resorts)."""
        height, width = self.config.display_size
        building_mask = np.zeros((height, width), dtype=np.float32)
        
        # Parameters
        cluster_count = kwargs.get('cluster_count', np.random.randint(1, 3))
        min_cluster_size = kwargs.get('min_cluster_size', 40)
        max_cluster_size = kwargs.get('max_cluster_size', 80)
        edge_buffer = kwargs.get('edge_buffer', 5)
        buildings_per_cluster = kwargs.get('buildings_per_cluster', (8, 20))
        
        for _ in range(cluster_count):
            cluster_w = np.random.randint(min_cluster_size, max_cluster_size + 1)
            cluster_h = np.random.randint(min_cluster_size, max_cluster_size + 1)
            
            if width <= edge_buffer * 2 + cluster_w or height <= edge_buffer * 2 + cluster_h:
                continue
                
            cluster_x = np.random.randint(edge_buffer, width - edge_buffer - cluster_w)
            cluster_y = np.random.randint(edge_buffer, height - edge_buffer - cluster_h)
            
            num_buildings = np.random.randint(buildings_per_cluster[0], buildings_per_cluster[1] + 1)
            buildings_placed = 0
            max_attempts = num_buildings * 20
            
            for attempt in range(max_attempts):
                if buildings_placed >= num_buildings:
                    break
                    
                building_w = np.random.randint(5, 20)
                building_h = np.random.randint(5, 20)
                
                if cluster_w <= building_w or cluster_h <= building_h:
                    continue
                    
                x = cluster_x + np.random.randint(0, cluster_w - building_w)
                y = cluster_y + np.random.randint(0, cluster_h - building_h)
                
                region = building_mask[y:y+building_h, x:x+building_w]
                overlap_ratio = np.sum(region > 0) / region.size
                if overlap_ratio > 0.3:
                    continue
                
                building_mask[y:y+building_h, x:x+building_w] = 1.0
                buildings_placed += 1
        
        return building_mask
    
    def _downsample_to_grid(self, display_layer: np.ndarray) -> np.ndarray:
        """
        Downsample high-resolution display layer to decision grid.
        
        Uses average pooling to maintain building density information
        at the decision grid resolution.
        """
        scale_h, scale_w = self.scale_factor
        grid_h, grid_w = self.config.grid_size
        
        # Reshape and average over each grid cell
        reshaped = display_layer.reshape(grid_h, scale_h, grid_w, scale_w)
        return np.mean(reshaped, axis=(1, 3))
    
    def _count_clusters(self, building_mask: np.ndarray) -> int:
        """Estimate number of building clusters using connected components."""
        try:
            from scipy import ndimage
            # Use connected components to count clusters
            labeled_array, num_clusters = ndimage.label(building_mask > 0)
            return num_clusters
        except ImportError:
            # Fallback if scipy not available
            return -1  # Indicate unknown
    
    def visualize_buildings(self, building_data: BuildingData, save_path: Optional[str] = None) -> None:
        """
        Visualize generated building distribution.
        
        Args:
            building_data: Generated building data
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Building Distribution - {building_data.metadata['pattern_type']}", y=0.98)
        
        # Display layer
        im1 = axes[0].imshow(building_data.display_layer, cmap='Greys', vmin=0, vmax=1)
        axes[0].set_title('Display Layer (1000×1000)', pad=15)
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        plt.colorbar(im1, ax=axes[0])
        
        # Decision grid  
        im2 = axes[1].imshow(building_data.decision_grid, cmap='Greys', vmin=0, vmax=1)
        axes[1].set_title('Decision Grid (50×50)', pad=15)
        axes[1].set_xlabel('Grid X')
        axes[1].set_ylabel('Grid Y')
        plt.colorbar(im2, ax=axes[1])
        
        # Add metadata text
        metadata_text = (f"Coverage: {building_data.metadata['coverage_ratio']:.1%}\n"
                        f"Clusters: {building_data.metadata['total_clusters']}\n"
                        f"Building pixels: {building_data.metadata['building_pixels']}")
        
        fig.text(0.02, 0.02, metadata_text, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Building visualization saved to: {save_path}")
        
        plt.show()
    
    def save_buildings(self, building_data: BuildingData, filepath: str) -> None:
        """Save building data to .npz file."""
        np.savez_compressed(
            filepath,
            display_layer=building_data.display_layer,
            decision_grid=building_data.decision_grid,
            metadata=building_data.metadata
        )
        print(f"Building data saved to: {filepath}")
    
    @staticmethod
    def load_buildings(filepath: str) -> BuildingData:
        """Load building data from .npz file."""
        data = np.load(filepath, allow_pickle=True)
        
        return BuildingData(
            display_layer=data['display_layer'],
            decision_grid=data['decision_grid'],
            metadata=data['metadata'].item()
        )


if __name__ == "__main__":
    # Example usage and testing
    generator = BuildingMapGenerator()
    
    print("BuildingMapGenerator initialized successfully!")
    print(f"Scale factor: {generator.scale_factor}")
    
    # Generate sparse building pattern
    buildings = generator.generate_buildings(BuildingPattern.SPARSE)
    print(f"Generated {buildings.metadata['total_clusters']} clusters")
    print(f"Building coverage: {buildings.metadata['coverage_ratio']:.1%}")