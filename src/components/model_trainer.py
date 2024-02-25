import os 
import sys
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, splitter, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_trainer_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting training and test")
            x_train, y_train, x_test, y_test = splitter(train_arr, test_arr)
            
            models= {
                "RandomForest" : RandomForestRegressor(),
                "DecisionTree" : DecisionTreeRegressor(),
                "AdaBoost" : AdaBoostRegressor()
                }
            
            params={
                "RandomForest":{
                # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
             
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,64,128,256]
            },
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                }
            logging.info('Declaring models and their parameters')
            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,
                                             models=models,param=params)
            
            logging.info('Model Evaluation has finished')
            
            best_model_score = sorted(model_report.values())[-1]
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            best_model.fit(x_train, y_train)
            logging.info('Got the best model')
            save_object(file_path = self.model_trainer_config.model_trainer_path,
                        obj = best_model)
            logging.info('Saved the best model')
            
            predicted_score = r2_score(y_test, best_model.predict(x_test))
            logging.info('Printeed the models performance on the test set')
            return predicted_score
                
        except Exception as e:
            raise CustomException(e, sys)
        