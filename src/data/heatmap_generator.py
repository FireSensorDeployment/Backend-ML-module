"""
Multi-layer static map generator with dual-resolution system.

This module provides HeatmapGenerator class for creating fire risk heatmaps
with support for multiple pattern types and dual-resolution architecture.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """Available pattern types for heatmap generation."""
    GAUSSIAN_HOTSPOT = "gaussian_hotspot"
    HORIZONTAL_GRADIENT = "horizontal_gradient" 
    VERTICAL_GRADIENT = "vertical_gradient"
    RADIAL_GRADIENT = "radial_gradient"
    RANDOM_NOISE = "random_noise"
    COMBINED = "combined"


@dataclass
class ResolutionConfig:
    """Configuration for dual-resolution system."""
    display_size: Tuple[int, int] = (1000, 1000)  # High-res for visualization
    grid_size: Tuple[int, int] = (50, 50)         # Low-res for RL training
    

@dataclass 
class HeatmapData:
    """Container for generated heatmap data."""
    display_layer: np.ndarray      # High-resolution display (1000x1000)
    decision_grid: np.ndarray      # RL decision grid (50x50)
    metadata: Dict[str, Any]       # Generation parameters and info


class HeatmapGenerator:
    """
    Multi-layer static map generator supporting dual-resolution system.
    
    Generates fire risk heatmaps with various patterns and maintains
    mathematically correct alignment between display and decision layers.
    """
    
    def __init__(self, resolution_config: ResolutionConfig = None):
        """Initialize the generator with resolution configuration."""
        self.config = resolution_config or ResolutionConfig()
        self._validate_resolution_config()
    
    def _validate_resolution_config(self):
        """Ensure resolution configuration is mathematically compatible."""
        display_h, display_w = self.config.display_size
        grid_h, grid_w = self.config.grid_size
        
        # Ensure even division for clean mapping
        if display_h % grid_h != 0 or display_w % grid_w != 0:
            raise ValueError(
                f"Display size {self.config.display_size} must be evenly divisible "
                f"by grid size {self.config.grid_size} for proper alignment"
            )
        
        self.scale_factor = (display_h // grid_h, display_w // grid_w)
    
    def _validate_grid_coords(self, grid_row: int, grid_col: int) -> bool:
        """Validate that grid coordinates are within valid bounds."""
        grid_h, grid_w = self.config.grid_size
        return 0 <= grid_row < grid_h and 0 <= grid_col < grid_w
    
    def _validate_display_coords(self, display_row: int, display_col: int) -> bool:
        """Validate that display coordinates are within valid bounds.""" 
        display_h, display_w = self.config.display_size
        return 0 <= display_row < display_h and 0 <= display_col < display_w

    def grid_to_display_coords(self, grid_row: int, grid_col: int, safe: bool = True) -> Tuple[int, int]:
        """
        Convert grid coordinates to display coordinates (top-left of cell).
        
        Args:
            grid_row: Grid row index
            grid_col: Grid column index  
            safe: If True, validate bounds and clamp to valid range
            
        Returns:
            Display coordinates tuple
        """
        if safe and not self._validate_grid_coords(grid_row, grid_col):
            # Clamp to nearest valid coordinate for RL safety
            grid_row = max(0, min(grid_row, self.config.grid_size[0] - 1))
            grid_col = max(0, min(grid_col, self.config.grid_size[1] - 1))
            
        display_row = grid_row * self.scale_factor[0]
        display_col = grid_col * self.scale_factor[1]
        return (display_row, display_col)
    
    def display_to_grid_coords(self, display_row: int, display_col: int, safe: bool = True) -> Tuple[int, int]:
        """
        Convert display coordinates to grid coordinates.
        
        Args:
            display_row: Display row coordinate
            display_col: Display column coordinate
            safe: If True, validate bounds and clamp to valid range
            
        Returns:
            Grid coordinates tuple
        """
        if safe and not self._validate_display_coords(display_row, display_col):
            # Clamp to nearest valid coordinate for RL safety
            display_row = max(0, min(display_row, self.config.display_size[0] - 1))
            display_col = max(0, min(display_col, self.config.display_size[1] - 1))
            
        grid_row = display_row // self.scale_factor[0] 
        grid_col = display_col // self.scale_factor[1]
        return (grid_row, grid_col)

    def get_grid_cell_center(self, grid_row: int, grid_col: int, safe: bool = True) -> Tuple[int, int]:
        """
        Get the center coordinates of a grid cell in display space.
        
        Args:
            grid_row: Grid row index
            grid_col: Grid column index
            safe: If True, validate bounds and clamp to valid range
            
        Returns:
            Display coordinates of the grid cell center
        """
        top_left = self.grid_to_display_coords(grid_row, grid_col, safe=safe)
        center_offset_h = self.scale_factor[0] // 2
        center_offset_w = self.scale_factor[1] // 2
        return (top_left[0] + center_offset_h, top_left[1] + center_offset_w)

    def get_grid_cell_region(self, grid_row: int, grid_col: int, safe: bool = True) -> Tuple[slice, slice]:
        """
        Get the display region (as slices) corresponding to a grid cell.
        
        Args:
            grid_row: Grid row index
            grid_col: Grid column index
            safe: If True, validate bounds and clamp to valid range
            
        Returns:
            Tuple of (row_slice, col_slice) for indexing display arrays
        """
        top_left = self.grid_to_display_coords(grid_row, grid_col, safe=safe)
        row_slice = slice(top_left[0], top_left[0] + self.scale_factor[0])
        col_slice = slice(top_left[1], top_left[1] + self.scale_factor[1])
        return (row_slice, col_slice)

    def generate_heatmap(self, pattern_type: PatternType, randomize: bool = False, **kwargs) -> HeatmapData:
        """
        Generate heatmap with specified pattern type.
        
        Args:
            pattern_type: Type of pattern to generate
            randomize: If True, randomize pattern parameters for training variety
            **kwargs: Pattern-specific parameters
            
        Returns:
            HeatmapData containing all generated layers
        """
        # Randomize parameters if requested
        if randomize:
            kwargs = self._randomize_parameters(pattern_type, **kwargs)
        
        # Generate high-resolution display layer first
        display_layer = self._generate_pattern(pattern_type, self.config.display_size, **kwargs)
        
        # Downsample to create decision grid
        decision_grid = self._downsample_to_grid(display_layer)
        
        # Note: Feature extraction removed - should be handled by separate modules
        
        # Note: region_mask removed - was unused placeholder functionality
        
        # Prepare metadata
        metadata = {
            'pattern_type': pattern_type.value,
            'resolution_config': self.config,
            'generation_params': kwargs,
            'scale_factor': self.scale_factor
        }
        
        return HeatmapData(
            display_layer=display_layer,
            decision_grid=decision_grid,
            metadata=metadata
        )
    
    def _randomize_parameters(self, pattern_type: PatternType, **kwargs) -> Dict[str, Any]:
        """
        Generate randomized parameters for pattern generation.
        
        Args:
            pattern_type: Type of pattern to randomize
            **kwargs: Existing parameters (will be overridden by random ones)
            
        Returns:
            Dictionary of randomized parameters
        """
        import random
        
        height, width = self.config.display_size
        
        if pattern_type == PatternType.GAUSSIAN_HOTSPOT:
            # Random number of hotspots (2-5)
            num_hotspots = random.randint(2, 5)
            
            # Random hotspot positions (avoid edges)
            edge_buffer = 100
            hotspot_centers = []
            for _ in range(num_hotspots):
                center_y = random.randint(edge_buffer, height - edge_buffer)
                center_x = random.randint(edge_buffer, width - edge_buffer)
                hotspot_centers.append((center_y, center_x))
            
            # Random intensities
            intensities = [random.uniform(0.3, 1.0) for _ in range(num_hotspots)]
            
            # Random sigma values
            sigmas = [random.uniform(30, 120) for _ in range(num_hotspots)]
            
            kwargs.update({
                'hotspot_centers': hotspot_centers,
                'intensities': intensities,
                'sigmas': sigmas
            })
            
        elif pattern_type == PatternType.HORIZONTAL_GRADIENT:
            kwargs.update({
                'direction': random.choice(['left_to_right', 'right_to_left']),
                'min_value': random.uniform(0.0, 0.3),
                'max_value': random.uniform(0.7, 1.0)
            })
            
        elif pattern_type == PatternType.VERTICAL_GRADIENT:
            kwargs.update({
                'direction': random.choice(['top_to_bottom', 'bottom_to_top']),
                'min_value': random.uniform(0.0, 0.3),
                'max_value': random.uniform(0.7, 1.0)
            })
            
        elif pattern_type == PatternType.RADIAL_GRADIENT:
            # Random center position
            center_y = random.randint(height // 4, 3 * height // 4)
            center_x = random.randint(width // 4, 3 * width // 4)
            
            kwargs.update({
                'center': (center_y, center_x),
                'direction': random.choice(['center_outward', 'outward_center']),
                'min_value': random.uniform(0.0, 0.3),
                'max_value': random.uniform(0.7, 1.0),
                'max_radius': random.uniform(200, 500)
            })
        
        return kwargs
    
    def _generate_pattern(self, pattern_type: PatternType, size: Tuple[int, int], **kwargs) -> np.ndarray:
        """Generate specific pattern type at given resolution."""
        # Note: size parameter reserved for future use with variable resolution patterns
        if pattern_type == PatternType.GAUSSIAN_HOTSPOT:
            return self._generate_gaussian_hotspot(**kwargs)
        elif pattern_type == PatternType.HORIZONTAL_GRADIENT:
            return self._generate_horizontal_gradient(**kwargs)
        elif pattern_type == PatternType.VERTICAL_GRADIENT:
            return self._generate_vertical_gradient(**kwargs)
        elif pattern_type == PatternType.RADIAL_GRADIENT:
            return self._generate_radial_gradient(**kwargs)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # TODO(human) - Pattern generation methods will be implemented here
    def _generate_gaussian_hotspot(self, **kwargs) -> np.ndarray:
        """Generate Gaussian hotspot pattern for fire risk simulation."""
        height, width = self.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # default primeters
        hotspot_centers = kwargs.get('hotspot_centers', [
            (int(height * 0.3), int(width * 0.3)),
            (int(height * 0.7), int(width * 0.7))
        ])
        intensities = kwargs.get('intensities', [0.8, 0.6])
        sigma = kwargs.get('sigma', min(height, width) * 0.1)
        sigmas = kwargs.get('sigmas', None)  # Support multiple sigma values
        
        if len(intensities) != len(hotspot_centers):
            intensities = [intensities[0]] * len(hotspot_centers) if len(intensities) == 1 else intensities[:len(hotspot_centers)]
        
        # Handle multiple sigma values or use single sigma for all
        if sigmas is not None:
            if len(sigmas) != len(hotspot_centers):
                sigmas = [sigmas[0]] * len(hotspot_centers) if len(sigmas) == 1 else sigmas[:len(hotspot_centers)]
        else:
            sigmas = [sigma] * len(hotspot_centers)
        
        y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
        for i in range(len(hotspot_centers)):
            center_row, center_col = hotspot_centers[i]
            intensity = intensities[i]
            current_sigma = sigmas[i]                   
            distance_sq = (x_coords - center_col)**2 + (y_coords - center_row)**2
            gaussian = intensity * np.exp(-distance_sq / (2 * current_sigma**2))
            heatmap += gaussian
        
        return heatmap
    
    def _generate_horizontal_gradient(self, **kwargs) -> np.ndarray:
        """Generate horizontal gradient pattern (left to right or right to left)."""
        height, width = self.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        direction = kwargs.get('direction', 'left_to_right')
        min_value = kwargs.get('min_value', 0.1)
        max_value = kwargs.get('max_value', 0.9)
        
        gradient_1d = np.linspace(min_value, max_value, width)
        if direction == "right_to_left":
            gradient_1d = gradient_1d[::-1]
        heatmap = np.broadcast_to(gradient_1d, (height, width))            
        
        return heatmap
    
    def _generate_vertical_gradient(self, **kwargs) -> np.ndarray:
        """Generate vertical gradient pattern (top to bottom or bottom to top)."""
        height, width = self.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # 默认参数
        direction = kwargs.get('direction', 'top_to_bottom')
        min_value = kwargs.get('min_value', 0.1)
        max_value = kwargs.get('max_value', 0.9)
        
        gradient_1d = np.linspace(min_value, max_value, height)
        if direction == "bottom_to_top":
            gradient_1d = gradient_1d[::-1]
        heatmap = np.broadcast_to(gradient_1d.reshape(-1, 1), (height, width))
        
        return heatmap
        
    def _generate_radial_gradient(self, **kwargs) -> np.ndarray:
        """Generate radial gradient pattern (center outward or inward)."""
        height, width = self.config.display_size
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # 默认参数
        center = kwargs.get('center', (height // 2, width // 2)) 
        center_y, center_x = center 
        # 中心点坐标
        direction = kwargs.get('direction', 'center_outward')     # 'center_outward' 或 'outward_center'
        min_value = kwargs.get('min_value', 0.1)
        max_value = kwargs.get('max_value', 0.9)
        max_radius = kwargs.get('max_radius', min(height, width) // 2)  # 最大半径
        
        y_coords, x_coords = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing='ij'
        )

        distance = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        normalized_dist = distance / max_radius

        normalized_dist = np.clip(normalized_dist, 0, 1)

        if direction == "center_outward":
            heatmap = min_value + normalized_dist * (max_value - min_value)
        else:
            heatmap = max_value - normalized_dist * (max_value - min_value)
        
        return heatmap
    
    def _downsample_to_grid(self, display_layer: np.ndarray) -> np.ndarray:
        """
        Downsample high-resolution display layer to decision grid.
        
        Uses average pooling for smooth downsampling while preserving
        overall heat distribution characteristics.
        """
        scale_h, scale_w = self.scale_factor
        grid_h, grid_w = self.config.grid_size
        
        # Reshape and average over each grid cell
        reshaped = display_layer.reshape(grid_h, scale_h, grid_w, scale_w)
        return np.mean(reshaped, axis=(1, 3))
    
    # Feature extraction methods removed - should be handled by separate modules
    # BuildingGenerator, RoadGenerator, etc. will handle semantic features
    
    def visualize_heatmap(self, heatmap_data: HeatmapData, save_path: Optional[str] = None) -> None:
        """
        Visualize generated heatmap data.
        
        Args:
            heatmap_data: Generated heatmap data
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Heatmap Visualization - {heatmap_data.metadata['pattern_type']}", y=0.98)
        
        # Display layer
        im1 = axes[0].imshow(heatmap_data.display_layer, cmap='hot')
        axes[0].set_title('Display Layer (1000×1000)', pad=15)
        axes[0].set_xlabel('X Coordinate')
        axes[0].set_ylabel('Y Coordinate')
        plt.colorbar(im1, ax=axes[0])
        
        # Decision grid
        im2 = axes[1].imshow(heatmap_data.decision_grid, cmap='hot')
        axes[1].set_title('Decision Grid (50×50)', pad=15)
        axes[1].set_xlabel('Grid X')
        axes[1].set_ylabel('Grid Y')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def save_heatmap(self, heatmap_data: HeatmapData, filepath: str) -> None:
        """Save heatmap data to .npy file."""
        np.savez_compressed(
            filepath,
            display_layer=heatmap_data.display_layer,
            decision_grid=heatmap_data.decision_grid,
            metadata=heatmap_data.metadata
        )
        print(f"Heatmap data saved to: {filepath}")
    
    @staticmethod
    def load_heatmap(filepath: str) -> HeatmapData:
        """Load heatmap data from .npy file."""
        data = np.load(filepath, allow_pickle=True)
        
        # Handle backward compatibility for files with region_mask
        if 'region_mask' in data.keys():
            print(f"Warning: Loaded file contains deprecated region_mask field (ignored)")
        
        return HeatmapData(
            display_layer=data['display_layer'],
            decision_grid=data['decision_grid'],
            metadata=data['metadata'].item()
        )


if __name__ == "__main__":
    # Example usage and testing
    generator = HeatmapGenerator()
    
    # This will be expanded with actual implementations
    print("HeatmapGenerator initialized successfully!")
    print(f"Scale factor: {generator.scale_factor}")