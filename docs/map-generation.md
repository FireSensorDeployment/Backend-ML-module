# Map Generation System Documentation

## Overview

HeatmapGenerator is a dual-resolution map generation system designed specifically for reinforcement learning (RL) sensor placement training. The system can generate various types of fire risk heatmaps and supports seamless conversion between high-precision display layers and low-precision decision grids.

## Core Features

### 🎯 Dual Resolution Architecture
- **Display Layer**: 1000×1000 pixel high-resolution map for visualization and detailed analysis
- **Decision Grid**: 50×50 grid low-resolution map for RL agent decision making
- **Scale Factor**: (20, 20) - each decision grid cell corresponds to a 20×20 pixel region

### 🌈 Diverse Pattern Generation
Supports 5 different risk distribution patterns:
- **Gaussian Hotspot**: Simulates fire high-risk areas
- **Horizontal Gradient**: Risk gradient from left-to-right or right-to-left
- **Vertical Gradient**: Risk gradient from top-to-bottom or bottom-to-top  
- **Radial Gradient**: Risk distribution from center-outward or outward-to-center
- **Combined Pattern**: Composite effects of multiple patterns

### 🔄 Intelligent Coordinate Conversion
- **Bidirectional Mapping**: Precise conversion between grid coordinates ↔ display coordinates
- **Boundary Validation**: Automatic boundary checking and safe clamping
- **Center Alignment**: Support for precise grid center positioning

## System Architecture

```
Input Parameters
    ↓
Pattern Generator → High-Resolution Display Layer (1000×1000)
    ↓                    ↓
Downsampler          Coordinate Converter
    ↓                    ↓
Decision Grid (50×50) ← → Helper Functions
    ↓
Metadata + Persistence
```

## Technical Specifications

### Resolution Configuration
```python
ResolutionConfig(
    display_size=(1000, 1000),  # High-resolution display layer
    grid_size=(50, 50)          # RL decision grid
)
```

### Data Structure
```python
HeatmapData {
    display_layer: np.ndarray,    # (1000, 1000) display layer
    decision_grid: np.ndarray,    # (50, 50) decision grid
    metadata: Dict[str, Any]      # generation parameters and config info
}
```

## Core API Reference

### Main Classes and Methods

#### HeatmapGenerator
The primary map generator class.

```python
generator = HeatmapGenerator(resolution_config=None)
```

**Parameters**:
- `resolution_config`: Optional resolution configuration object

#### Map Generation
```python
def generate_heatmap(pattern_type: PatternType, **kwargs) -> HeatmapData
```

Generate a heatmap of the specified type.

**Parameters**:
- `pattern_type`: Pattern type enumeration
- `**kwargs`: Pattern-specific parameters

**Returns**: HeatmapData object containing all generated layers

### Coordinate Conversion Methods

#### Grid to Display Coordinates
```python  
def grid_to_display_coords(grid_row: int, grid_col: int, safe: bool = True) -> Tuple[int, int]
```

Convert grid coordinates to display coordinates (top-left corner).

#### Display to Grid Coordinates  
```python
def display_to_grid_coords(display_row: int, display_col: int, safe: bool = True) -> Tuple[int, int]
```

Convert display coordinates to grid coordinates.

### Helper Function Methods

#### Get Grid Cell Center
```python
def get_grid_cell_center(grid_row: int, grid_col: int, safe: bool = True) -> Tuple[int, int]
```

Get the center coordinates of a grid cell in display space, used for precise sensor placement.

#### Get Grid Cell Region
```python  
def get_grid_cell_region(grid_row: int, grid_col: int, safe: bool = True) -> Tuple[slice, slice]
```

Get the display region slices corresponding to a grid cell, used for batch operations.

### Data Persistence

#### Save Map
```python
def save_heatmap(heatmap_data: HeatmapData, filepath: str) -> None
```

#### Load Map
```python
@staticmethod  
def load_heatmap(filepath: str) -> HeatmapData
```

## Pattern Detailed Description

### 1. Gaussian Hotspot Pattern
Simulates realistic fire high-risk area distributions.

**Key Parameters**:
- `hotspot_centers`: List of hotspot center coordinates
- `intensities`: List of corresponding intensity values
- `sigma`: Standard deviation of Gaussian distribution (controls spread range)
- `sigmas`: Different standard deviations for multiple hotspots

**Usage Example**:
```python
heatmap = generator.generate_heatmap(
    PatternType.GAUSSIAN_HOTSPOT,
    hotspot_centers=[(300, 700), (700, 300)],
    intensities=[0.9, 0.7],
    sigma=80
)
```

### 2. Horizontal Gradient Pattern
Creates risk gradients from left-to-right or right-to-left.

**Key Parameters**:
- `direction`: 'left_to_right' or 'right_to_left'
- `min_value`: Minimum risk value
- `max_value`: Maximum risk value

### 3. Vertical Gradient Pattern
Creates risk gradients from top-to-bottom or bottom-to-top.

**Key Parameters**:
- `direction`: 'top_to_bottom' or 'bottom_to_top'  
- `min_value`: Minimum risk value
- `max_value`: Maximum risk value

### 4. Radial Gradient Pattern
Creates circular risk distribution patterns.

**Key Parameters**:
- `center`: Center coordinates
- `direction`: 'center_outward' or 'outward_center'
- `min_value`: Minimum risk value
- `max_value`: Maximum risk value  
- `max_radius`: Maximum radius

## Downsampling Mechanism

The system uses **average pooling** to downsample the high-precision display layer to the decision grid:

```python
def _downsample_to_grid(display_layer):
    # Reshape 1000×1000 to 50×20×50×20
    reshaped = display_layer.reshape(grid_h, scale_h, grid_w, scale_w)
    # Calculate average value for each grid cell
    return np.mean(reshaped, axis=(1, 3))
```

**Advantages**:
- Preserves overall risk distribution characteristics
- High computational efficiency
- Numerical stability suitable for RL training

## Boundary Handling and Safety

### Coordinate Validation
All coordinate conversion methods include optional safety mode:

```python
if safe and not self._validate_grid_coords(grid_row, grid_col):
    # Automatically clamp to valid range
    grid_row = max(0, min(grid_row, self.config.grid_size[0] - 1))
    grid_col = max(0, min(grid_col, self.config.grid_size[1] - 1))
```

### Resolution Compatibility Check
Validates that display dimensions must be evenly divisible by grid dimensions during initialization:

```python
if display_h % grid_h != 0 or display_w % grid_w != 0:
    raise ValueError("Display size must be evenly divisible by grid size for proper alignment")
```

## Performance Characteristics

### Benchmark Performance
- **Generation Time**: ~0.024 seconds (1000×1000 map)
- **Memory Usage**: ~3.8MB (including all layers and metadata)
- **Conversion Speed**: Coordinate conversion <0.000001 seconds/call

### Optimization Recommendations
- Batch coordinate conversions are more efficient than single conversions
- Reuse generator instances with the same configuration
- Use `get_grid_cell_region()` for large batch operations

## Use Cases

### RL Environment Integration
```python
# 1. Generate training map
generator = HeatmapGenerator()
heatmap_data = generator.generate_heatmap(PatternType.GAUSSIAN_HOTSPOT)

# 2. RL agent works on decision grid
state = heatmap_data.decision_grid  # (50, 50)
action = rl_agent.select_action(state)

# 3. Map decisions back to display layer for visualization
for grid_pos in sensor_positions:
    center = generator.get_grid_cell_center(*grid_pos)
    # Mark sensor position on display layer
```

### Batch Map Generation
```python
# Generate multiple scenarios for training
scenarios = []
for pattern in [PatternType.GAUSSIAN_HOTSPOT, PatternType.RADIAL_GRADIENT]:
    for intensity in [0.5, 0.7, 0.9]:
        heatmap = generator.generate_heatmap(
            pattern,
            intensities=[intensity],
            sigma=60
        )
        scenarios.append(heatmap)
```

### Sensor Coverage Visualization
```python  
# Display sensor coverage areas on the map
display_layer = heatmap_data.display_layer.copy()

for sensor_pos in [(10, 15), (25, 30)]:
    # Highlight coverage area
    region = generator.get_grid_cell_region(*sensor_pos)
    display_layer[region] *= 1.3
    
    # Mark sensor center
    center = generator.get_grid_cell_center(*sensor_pos)
    display_layer[center[0]-2:center[0]+3, center[1]-2:center[1]+3] = 1.0
```

## Error Handling

### Common Errors and Solutions

**Configuration Error**:
```python
# Error: Incompatible dimension configuration
ValueError: Display size (1000, 1000) must be evenly divisible by grid size (33, 33)

# Solution: Use compatible dimensions
ResolutionConfig(display_size=(990, 990), grid_size=(33, 33))
```

**Coordinate Out of Bounds**:
```python
# Automatic handling: Use safe=True (default)
center = generator.get_grid_cell_center(60, 60, safe=True)  # Auto-clamps to (49, 49)
```

**Missing Pattern Parameters**:
```python
# Ensure necessary parameters are provided
heatmap = generator.generate_heatmap(
    PatternType.GAUSSIAN_HOTSPOT,
    hotspot_centers=[(500, 500)],  # Required
    intensities=[0.8]              # Required
)
```

## Extension and Customization

### Adding New Pattern Types
1. Extend the `PatternType` enumeration
2. Add new branch in `_generate_pattern`
3. Implement corresponding generation method

### Custom Resolution
```python
# High-resolution configuration
high_res_config = ResolutionConfig(
    display_size=(2000, 2000),
    grid_size=(100, 100)
)
generator = HeatmapGenerator(high_res_config)
```

### External Data Integration
```python
# Generate based on real geographic data
def create_from_gis_data(gis_data):
    # Process GIS data...
    display_layer = process_gis_to_array(gis_data)
    decision_grid = generator._downsample_to_grid(display_layer)
    return HeatmapData(display_layer, decision_grid, metadata)
```

## Test Coverage

The system includes a comprehensive test suite:

- **Initialization Tests**: Verify configuration correctness
- **Coordinate Mapping Tests**: Bidirectional conversion accuracy
- **Pattern Generation Tests**: Correctness of all pattern types
- **Boundary Case Tests**: Extreme values and boundary handling
- **Performance Tests**: Generation speed and memory usage
- **Persistence Tests**: Save/load functionality integrity

Run tests:
```bash
python tests/test_heatmap_generator.py
```

---

*Documentation Version: 1.0*  
*Last Updated: 2025-09-02*