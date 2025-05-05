import sys 
import os 
import pandas as pd 
from src.exception import CustomException
from src.logger import logging

from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info(f"This is the  data before scaling{features}")
            logging.info(f"This is the data with null values {features.isnull().sum()}")
            logging.info(f"the null data is features{features[features.isnull().any(axis=1)]}")
            logging.info(features['race_ethnicity'][0])
            logging.info(f"This is the  data after transforming {features.isnull().sum()}")  
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info("Error in prediction")
            raise CustomException(e, sys) from e
class CustomData:
    def __init__(self,
                    gender:str,
                    race_ethnicity:str,
                    parental_level_of_education:str,
                    lunch:str,
                    test_preparation_course:str,
                    writing_score:int,
                    reading_score:int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score
    

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "writing_score": [self.writing_score],
                "reading_score": [self.reading_score]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Custom data converted to DataFrame")
            return df
        except Exception as e:
            logging.info("Error while converting custom data to DataFrame")
            raise CustomException(e, sys) from e