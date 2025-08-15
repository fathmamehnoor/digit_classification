#!/usr/bin/env python3
"""
Setup script for Audio Digit Classification Project
Downloads trained models and sets up the environment
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_models():
    """Download trained models from a cloud storage"""
    print("🔧 Setting up Audio Digit Classification Project")
    print("=" * 50)
    
    # Create models/saved directory if it doesn't exist
    models_dir = Path("models/saved")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if models already exist
    model_files = [
        "best_model_2d.pth",
        "preprocessor_2d.pkl", 
        "scaler_2d.pkl"
    ]
    
    existing_files = [f for f in model_files if (models_dir / f).exists()]
    
    if existing_files:
        print(f"✅ Found existing model files: {', '.join(existing_files)}")
        print("Skipping download...")
        return True
    
    print("📥 Downloading trained models...")
    print("Note: This requires an internet connection")
    
    # For now, we'll provide instructions for manual download
    # In a real scenario, you would upload models to cloud storage
    print("\n📋 Manual Setup Instructions:")
    print("1. Train the model using: python models/train.py")
    print("2. The trained models will be saved in models/saved/")
    print("3. Required files:")
    for file in model_files:
        print(f"   - {file}")
    
    print("\n🚀 Quick Start:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Train the model: python models/train.py")
    print("3. Launch the app: python ui/gradio_app.py")
    
    return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        'torch', 'librosa', 'gradio', 'numpy', 
        'scikit-learn', 'matplotlib', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def main():
    """Main setup function"""
    print("🎤 Audio Digit Classification - Setup")
    print("=" * 40)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Download models
    models_ok = download_models()
    
    print("\n" + "=" * 40)
    if deps_ok and models_ok:
        print("✅ Setup complete! You can now run the application.")
        print("🚀 Launch with: python ui/gradio_app.py")
    else:
        print("⚠️  Setup incomplete. Please follow the instructions above.")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main()
