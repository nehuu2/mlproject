#!/usr/bin/env python3
"""
Deployment check script for the ML project
This script verifies that all components are working correctly
"""

import os
import sys
import pandas as pd
import numpy as np

def check_environment():
    """Check if all required packages are available"""
    print("=== Environment Check ===")
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'dill', 'flask', 
        'xgboost', 'catboost', 'seaborn', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        return False
    else:
        print("\n✓ All required packages are available")
        return True

def check_files():
    """Check if all required files exist"""
    print("\n=== File Check ===")
    required_files = [
        "artifacts/model.pkl",
        "artifacts/preprocessor.pkl",
        "src/utils.py",
        "src/exception.py",
        "src/pipepline/predict_pipeline.py",
        "app.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    else:
        print("\n✓ All required files exist")
        return True

def check_model():
    """Check if the model can be loaded and used"""
    print("\n=== Model Check ===")
    try:
        from src.utils import load_object
        
        # Load model and preprocessor
        model = load_object("artifacts/model.pkl")
        preprocessor = load_object("artifacts/preprocessor.pkl")
        print("✓ Model and preprocessor loaded successfully")
        
        # Test prediction
        test_data = pd.DataFrame({
            "gender": ["male"],
            "race_ethnicity": ["group A"],
            "parental_level_of_education": ["bachelor's degree"],
            "lunch": ["standard"],
            "test_preparation_course": ["none"],
            "reading_score": [75],
            "writing_score": [80]
        })
        
        # Transform and predict
        data_scaled = preprocessor.transform(test_data)
        prediction = model.predict(data_scaled)
        
        print(f"✓ Test prediction successful: {prediction[0]:.2f}")
        return True
        
    except Exception as e:
        print(f"✗ Model check failed: {str(e)}")
        return False

def check_flask_app():
    """Check if the Flask app can be imported"""
    print("\n=== Flask App Check ===")
    try:
        from app import app
        print("✓ Flask app imported successfully")
        return True
    except Exception as e:
        print(f"✗ Flask app import failed: {str(e)}")
        return False

def main():
    """Run all checks"""
    print("Starting deployment checks...\n")
    
    checks = [
        ("Environment", check_environment),
        ("Files", check_files),
        ("Model", check_model),
        ("Flask App", check_flask_app)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ {name} check failed with exception: {str(e)}")
            results.append((name, False))
    
    print("\n=== Summary ===")
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n✓ All checks passed! The application should work correctly.")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
