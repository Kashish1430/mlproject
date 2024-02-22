import os 
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_path: str=os.path.join('artifacts', 'train.csv')
    test_path: str=os.path.join('artifacts', 'test.csv')
    raw_path: str=os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self, ):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the Data Ingestion method')
        try:
            df = pd.read_csv('data\Anarcho_Monthly_Score.csv')
            logging.info('Read the data as a dataframe from csv')
            os.makedirs(os.path.dirname(self.ingestion_config.train_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path, index=False, header=True)
            logging.info('Train Test split has been initiated')
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state = 42)
            train_set.to_csv(self.ingestion_config.train_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_path, index=False, header=True)
            logging.info('Ingestion of data has been compelted')
            return (
                self.ingestion_config.train_path,
                self.ingestion_config.test_path
                )
        except Exception as e:
            raise CustomException(e, sys)
            
        
if __name__ == '__main__':
    dataingestion = DataIngestion()
    train, test = dataingestion.initiate_data_ingestion()
    
    data_transform = DataTransformation()
    train_arr, test_arr,_ = data_transform.initiate_data_transformation(train, test)
    modeltrain = ModelTrainer()
    score = modeltrain.initiate_model_training(train_arr, test_arr)
    print(score)