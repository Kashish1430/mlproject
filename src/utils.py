import os
import sys
import pickle
import numpy as np
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.logger import logging
import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file:
            return dill.load(file)
    
    
    except Exception as e:
        raise CustomException(e, sys)

def splitter(train, test):
    try:
        train = train[~np.isnan(train).any(axis=1)]
        test = test[~np.isnan(test).any(axis=1)]
        x_train = train[:, :-1]
        y_train = train[:, -1]
        x_test = test[:, :-1]
        y_test = test[:, -1]
        
        return x_train, y_train, x_test, y_test
    
    except Exception as e:
        raise CustomException(e, sys)
        
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    logging.info('The function from utils has started')
    try:
        report = {}

        for i in range(len(list(models))):
            logging.info("Starting the loop")
            model = list(models.values())[i]
            logging.info('Got The model')
            para=param[list(models.keys())[i]]
            logging.info('Got the parameters')
            gs = GridSearchCV(model,para,cv=3)
            logging.info('created GS obj')
            gs.fit(X_train,y_train)
            logging.info('Fitting Gs')
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
            logging.info('Function is over')

        return report

    except Exception as e:
        raise CustomException(e, sys)
    