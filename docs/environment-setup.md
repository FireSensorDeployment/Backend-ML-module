  ## Environment Setup

  ### Requirements
  - Python 3.12+
  - Virtual environment (conda, venv, or your preference)

  ### Installation
  1. Create your virtual environment:
     ```bash
     # Using venv
     python -m venv venv
     source venv/bin/activate  # Linux/Mac
     # or
     .\venv\Scripts\activate  # Windows

     # OR using conda
     conda create -n SNProject python=3.12
     conda activate SNProject

  2. Install dependencies:
  pip install -r requirements.txt
  3. Verify installation:
  python test_env.py

  ### requirements.txt
  ```txt
# Reinforcement Learning
stable-baselines3
gymnasium

# Core ML/Data Science
numpy

# Computer Vision & Visualization
opencv-python
matplotlib