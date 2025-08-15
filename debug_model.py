#!/usr/bin/env python3
"""
Debug script to identify model loading issues
"""

import os
import sys
import traceback
import pandas as pd

def debug_model_loading():
    """Debug model loading step by step"""
    print("=== Model Loading Debug ===")
    
    try:
        # Step 1: Check current directory and artifacts
        print(f"1. Current working directory: {os.getcwd()}")
        print(f"2. Artifacts directory exists: {os.path.exists('artifacts')}")
        
        if os.path.exists('artifacts'):
            print(f"3. Artifacts contents: {os.listdir('artifacts')}")
        
        # Step 2: Check model files
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        
        print(f"4. Model path: {model_path}")
        print(f"5. Model exists: {os.path.exists(model_path)}")
        print(f"6. Preprocessor path: {preprocessor_path}")
        print(f"7. Preprocessor exists: {os.path.exists(preprocessor_path)}")
        
        if os.path.exists(model_path):
            print(f"8. Model file size: {os.path.getsize(model_path)} bytes")
        if os.path.exists(preprocessor_path):
            print(f"9. Preprocessor file size: {os.path.getsize(preprocessor_path)} bytes")
        
        # Step 3: Try to import utils
        print("10. Trying to import utils...")
        from src.utils import load_object
        print("✓ Utils imported successfully")
        
        # Step 4: Try to load model
        print("11. Trying to load model...")
        model = load_object(model_path)
        print("✓ Model loaded successfully")
        print(f"12. Model type: {type(model)}")
        
        # Step 5: Try to load preprocessor
        print("13. Trying to load preprocessor...")
        preprocessor = load_object(preprocessor_path)
        print("✓ Preprocessor loaded successfully")
        print(f"14. Preprocessor type: {type(preprocessor)}")
        
        # Step 6: Test with sample data
        print("15. Testing with sample data...")
        sample_data = pd.DataFrame({
            "gender": ["male"],
            "race_ethnicity": ["group A"],
            "parental_level_of_education": ["bachelor's degree"],
            "lunch": ["standard"],
            "test_preparation_course": ["none"],
            "reading_score": [75],
            "writing_score": [80]
        })
        
        print(f"16. Sample data shape: {sample_data.shape}")
        print(f"17. Sample data columns: {sample_data.columns.tolist()}")
        
        # Step 7: Transform data
        print("18. Transforming data...")
        data_scaled = preprocessor.transform(sample_data)
        print(f"19. Scaled data shape: {data_scaled.shape}")
        
        # Step 8: Make prediction
        print("20. Making prediction...")
        prediction = model.predict(data_scaled)
        print(f"21. Prediction: {prediction[0]}")
        
        print("\n✓ All steps completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        print(f"Full traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = debug_model_loading()
    if not success:
        sys.exit(1)
