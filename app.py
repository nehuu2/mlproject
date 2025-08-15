from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import os
import logging

from sklearn.preprocessing import StandardScaler
from src.pipepline.predict_pipeline import CustomData,PredictPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

application=Flask(__name__)

app=application

##Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health_check():
    try:
        logger.info("Health check started")
        
        # Test if model files exist
        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        
        model_exists = os.path.exists(model_path)
        preprocessor_exists = os.path.exists(preprocessor_path)
        
        logger.info(f"Model exists: {model_exists}, Preprocessor exists: {preprocessor_exists}")
        
        # Test if model can be loaded
        if model_exists and preprocessor_exists:
            try:
                from src.utils import load_object
                model = load_object(model_path)
                preprocessor = load_object(preprocessor_path)
                model_loaded = True
                logger.info("Model and preprocessor loaded successfully")
            except Exception as load_error:
                logger.error(f"Error loading model: {str(load_error)}")
                model_loaded = False
        else:
            model_loaded = False
            
        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_exists": model_exists,
            "preprocessor_exists": preprocessor_exists,
            "model_loaded": model_loaded,
            "current_directory": os.getcwd(),
            "artifacts_directory": os.path.exists("artifacts"),
            "artifacts_contents": os.listdir("artifacts") if os.path.exists("artifacts") else []
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "current_directory": os.getcwd()
        }

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            logger.info("Prediction request received")
            
            # Validate form data
            gender = request.form.get('gender')
            ethnicity = request.form.get('ethnicity')
            parental_level_of_education = request.form.get('parental_level_of_education')
            lunch = request.form.get('lunch')
            test_preparation_course = request.form.get('test_preparation_course')
            writing_score = request.form.get('writing_score')
            reading_score = request.form.get('reading_score')
            
            logger.info(f"Form data: gender={gender}, ethnicity={ethnicity}, parental_education={parental_level_of_education}, lunch={lunch}, test_course={test_preparation_course}, writing={writing_score}, reading={reading_score}")
            
            # Check if all required fields are present
            if not all([gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, writing_score, reading_score]):
                return render_template('home.html', results="Error: All fields are required")
            
            # Validate numeric scores
            try:
                writing_score = float(writing_score)
                reading_score = float(reading_score)
            except ValueError:
                return render_template('home.html', results="Error: Reading and writing scores must be valid numbers")
            
            data = CustomData(
                gender=gender,
                race_ethnicity=ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                writing_score=writing_score,
                reading_score=reading_score
            )
            pred_df = data.get_data_as_data_frame()
            logger.info(f"DataFrame created: {pred_df.shape}")

            predict_pipeline=PredictPipeline()
            logger.info("Starting prediction...")
            results=predict_pipeline.predict(pred_df)
            prediction_value = float(results[0])  # Convert numpy.float64 to Python float
            logger.info(f"Prediction completed: {prediction_value}")
            return render_template('home.html',results=prediction_value)
        except FileNotFoundError as e:
            logger.error(f"File not found error: {str(e)}")
            return render_template('home.html', results=f"Error: Model files not found. Please check deployment.")
        except ValueError as e:
            logger.error(f"Value error: {str(e)}")
            return render_template('home.html', results=f"Error: Invalid input data - {str(e)}")
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            return render_template('home.html', results=f"Error: Missing dependencies - {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return render_template('home.html', results=f"Error: An unexpected error occurred - {str(e)}")
    
##if __name__=="__main__":
  ##  app.run(host="0.0.0.0",debug=True)    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
