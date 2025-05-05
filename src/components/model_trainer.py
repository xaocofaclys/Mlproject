import os 
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.utils import eval_reg_model, evaluate_models, hyperparameter_tuning
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging  


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    model_report_file_path = os.path.join('artifacts', 'model_report.txt')
    best_model_file_path = os.path.join('artifacts', 'best_model.pkl')



class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.models = {
            'RandomForestRegressor': RandomForestRegressor(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'LinearRegression': LinearRegression(),
            'LogisticRegression': LogisticRegression(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'GradientBoostingRegressor': GradientBoostingRegressor(),
            'XGBRegressor': XGBRegressor(),
            'AdaBoostRegressor': AdaBoostRegressor(),
            'CatBoostRegressor': CatBoostRegressor()
        }

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1],
                test_array[:, :-1], 
                test_array[:, -1]
            )

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = self.models)
            logging.info(f"Model report: {model_report}")
            best_model_score = max([metrics['R2'] for metrics in model_report.values()])
            best_model_name = [model for model, metrics in model_report.items() if metrics['R2'] == best_model_score][0]
            print(f"Best model: {best_model_name} with R2 score: {best_model_score}")
            logging.info(f"Best model: {best_model_name} with R2 score: {best_model_score}")
            save_object(
                obj = self.models[best_model_name],
                file_path = self.model_trainer_config.trained_model_file_path
            )
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def hyperparameter_tuning(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data for hyperparameter tuning")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1],
                test_array[:, :-1], 
                test_array[:, -1]
            )

            param_grid_for_Linear_regression = {
                'fit_intercept': [True, False],
                'positive': [True, False],
                'copy_X': [True, False],
                'n_jobs': [1, 2, 3]
            }

            best_model = hyperparameter_tuning(
                model = LinearRegression(),
                X_train = X_train,
                y_train = y_train,
                param_grid = param_grid_for_Linear_regression
            )

            report = eval_reg_model(
                model = best_model,
                X_test = X_test,
                y_test = y_test
            )
            save_object(
                obj = best_model,
                file_path = self.model_trainer_config.best_model_file_path
            )
            
            logging.info(f"Best model after hyperparameter tuning: {best_model} with score: {report}")
        except Exception as e:
            raise CustomException(e, sys) from e