import os
import sys
import pandas as pd
from src.utils import load_object

def test_model_loading():
    try:
        # Test model loading
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        
        print(f"Testing model loading...")
        print(f"Model path: {model_path}")
        print(f"Preprocessor path: {preprocessor_path}")
        print(f"Model exists: {os.path.exists(model_path)}")
        print(f"Preprocessor exists: {os.path.exists(preprocessor_path)}")
        
        if not os.path.exists(model_path):
            print("ERROR: Model file not found!")
            return False
            
        if not os.path.exists(preprocessor_path):
            print("ERROR: Preprocessor file not found!")
            return False
        
        # Load model and preprocessor
        model = load_object(model_path)
        preprocessor = load_object(preprocessor_path)
        
        print("✓ Model and preprocessor loaded successfully")
        
        # Test with sample data
        sample_data = pd.DataFrame({
            "gender": ["male"],
            "race_ethnicity": ["group A"],
            "parental_level_of_education": ["bachelor's degree"],
            "lunch": ["standard"],
            "test_preparation_course": ["none"],
            "reading_score": [75],
            "writing_score": [80]
        })
        
        print(f"Sample data shape: {sample_data.shape}")
        print(f"Sample data columns: {sample_data.columns.tolist()}")
        
        # Transform and predict
        data_scaled = preprocessor.transform(sample_data)
        prediction = model.predict(data_scaled)
        
        print(f"✓ Prediction successful: {prediction[0]}")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Tests failed!")
        sys.exit(1)
