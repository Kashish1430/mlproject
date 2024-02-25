import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipleline():
    def __init__(self):
        pass
    
    def predict(self, features):
        try: 
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_transformed = preprocessor.transform(features)
            preds = model.predict(data_transformed)
            return preds
         
        except Exception as e:
            raise CustomException(e, sys)
            
class CustomData():
    def __init__(self,
                 year:int,
                 month:int,
                 toxicity:int,
                 Fulldate:int,
                 label:str):
        self.year = year
        self.month = month
        self.toxicity = toxicity
        self.Fulldate = Fulldate
        self.label = label
        
    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({
                "year":[self.year],
                "month":[self.month],
                "toxicity":[self.toxicity],
                "Full Date":[self.Fulldate],
                "label":[self.label]})
        except Exception as e:
            raise CustomException(e, sys)
            
    