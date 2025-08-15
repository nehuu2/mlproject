import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
          try:
            import traceback
            import logging
            
            logger = logging.getLogger(__name__)
            ##this is for my personal pc 
            ##model_path='artifacts\model.pkl'
            ##this is for deployment on render
            
            # Try multiple possible paths for deployment
            possible_paths = [
                os.path.join("artifacts", "model.pkl"),
                os.path.join(os.getcwd(), "artifacts", "model.pkl"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "artifacts", "model.pkl")
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
                    
            if model_path is None:
                raise FileNotFoundError(f"Model file not found. Tried paths: {possible_paths}")
                
            preprocessor_path = model_path.replace("model.pkl", "preprocessor.pkl")
            
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at: {preprocessor_path}")
            
            # Check file sizes
            model_size = os.path.getsize(model_path)
            preprocessor_size = os.path.getsize(preprocessor_path)
            
            logger.info(f"Loading model from: {model_path} (size: {model_size} bytes)")
            logger.info(f"Loading preprocessor from: {preprocessor_path} (size: {preprocessor_size} bytes)")
            
            if model_size == 0:
                raise ValueError("Model file is empty")
            if preprocessor_size == 0:
                raise ValueError("Preprocessor file is empty")
            
            model = load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            
            logger.info("Model and preprocessor loaded successfully")
            logger.info(f"Input features shape: {features.shape}")
            logger.info(f"Input features columns: {features.columns.tolist()}")
            logger.info(f"Input features data types: {features.dtypes}")
            
            data_scaled=preprocessor.transform(features)
            logger.info(f"Scaled data shape: {data_scaled.shape}")
            
            preds = model.predict(data_scaled)
            logger.info(f"Predictions shape: {preds.shape}")
            logger.info(f"Predictions: {preds}")
            return preds
          except Exception as e:
            logger.error(f"Error in predict method: {str(e)}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Model path exists: {os.path.exists(model_path) if 'model_path' in locals() else 'N/A'}")
            logger.error(f"Preprocessor path exists: {os.path.exists(preprocessor_path) if 'preprocessor_path' in locals() else 'N/A'}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise CustomException(e,sys) 



class CustomData:
    def __init__(self,
                 gender:str,
                 race_ethnicity:str,
                 parental_level_of_education,
                 lunch:str,
                 test_preparation_course:str,
                 reading_score:int,
                 writing_score:int):

                self.gender = gender
                self.race_ethnicity = race_ethnicity
                self.parental_level_of_education = parental_level_of_education
                self.lunch = lunch
                self.test_preparation_course = test_preparation_course
                self.reading_score = reading_score
                self.writing_score = writing_score   
    
    def get_data_as_data_frame(self):
          try:
            import logging
            logger = logging.getLogger(__name__)
            
            # Validate input data
            if not all([self.gender, self.race_ethnicity, self.parental_level_of_education, 
                       self.lunch, self.test_preparation_course]):
                raise ValueError("All categorical fields must be provided")
            
            if not isinstance(self.reading_score, (int, float)) or not isinstance(self.writing_score, (int, float)):
                raise ValueError("Reading and writing scores must be numeric")
            
            if self.reading_score < 0 or self.reading_score > 100 or self.writing_score < 0 or self.writing_score > 100:
                raise ValueError("Scores must be between 0 and 100")
            
            custom_data_input_dict = {
                  "gender": [self.gender],
                  "race_ethnicity":[self.race_ethnicity],
                  "parental_level_of_education":[self.parental_level_of_education],
                  "lunch":[self.lunch],
                  "test_preparation_course":[self.test_preparation_course],
                  "reading_score":[self.reading_score],
                  "writing_score":[self.writing_score],
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logger.info(f"Created DataFrame with shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            logger.info(f"DataFrame data types: {df.dtypes}")
            return df
          except Exception as e:
                logger.error(f"Error in get_data_as_data_frame: {str(e)}")
                raise CustomException(e,sys)

            


        
        

