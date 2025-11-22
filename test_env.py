"""Test if all required packages are installed correctly."""

def test_imports():
    """Test all required imports."""
    # 包名映射：显示名称 -> 实际导入名
    packages = {
        'stable_baselines3': 'stable_baselines3',
        'gymnasium': 'gymnasium',
        'numpy': 'numpy',
        'opencv-python': 'cv2',  # 特殊映射
        'matplotlib': 'matplotlib'
    }

    for display_name, import_name in packages.items():
        try:
            __import__(import_name)
            print(f"✓ {display_name}")
        except ImportError:
            print(f"✗ {display_name} - Please install with: pip install {display_name}")
            return False

    print("\n✅ All packages installed successfully!")
    return True

if __name__ == "__main__":
    import sys
    if not test_imports():
        sys.exit(1)