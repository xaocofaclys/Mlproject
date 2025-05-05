import os 
import sys 
import numpy as np 
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

def save_object(obj, file_path):
    """
    This function saves the given object to the specified file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e
    

def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    This function evaluates the performance of different regression models on the given training and testing data.
    It returns a dictionary containing the model names and their respective evaluation metrics.
    """
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            report[model_name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
        return report
    except Exception as e:
        raise CustomException(e, sys) from e
    


def hyperparameter_tuning(model, X_train, y_train, param_grid):
    """
    This function performs hyperparameter tuning using GridSearchCV for the given model and training data.
    It returns the best model after tuning.
    """
    try:
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        return best_model
    except Exception as e:
        raise CustomException(e, sys) from e
    
def eval_reg_model(model, X_test, y_test):
    """
    This function evaluates the performance of the given model on the testing data.
    It returns the evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    except Exception as e:
        raise CustomException(e, sys) from e
    

def load_object(file_path):
    """
    This function loads the object from the specified file path using pickle.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e