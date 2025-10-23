import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            logging.info("Prediction process has started")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            logging.info("Preprocessor and model have been loaded")
            features_encoded=preprocessor.transform(features)
            feature_predicted=model.predict(features_encoded)
            return feature_predicted
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    
    def get_data_as_data_frame(self):
        try:
            data_dict={"gender":[self.gender],"race_ethnicity":[self.race_ethnicity],"parental_level_of_education":[self.parental_level_of_education],"lunch":[self.lunch],"test_preparation_course":[self.test_preparation_course],"reading_score":[self.reading_score],"writing_score":[self.writing_score]}
            df=pd.DataFrame(data_dict)
            return df
        except Exception as e:
            raise CustomException(e,sys)
            