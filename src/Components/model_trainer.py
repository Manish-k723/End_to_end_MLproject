import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Lasso
)
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import (
    save_obj,
    evaluate_model
)

@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() #Inside this variable, we have stored the path.
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Spliting training and test input data")

            X_train, y_train, X_test, y_test = (
            train_arr[:,:-1], train_arr[:,-1],
            test_arr[:,:-1], test_arr[:,-1]
            )
            models = {
            'Linear Regression':LinearRegression(),
            'Lasso Regression':Lasso(),
            'K-Nearest Neighbors': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost Regressor':XGBRegressor(),
            'Catboost Regressor': CatBoostRegressor(verbose = False),
            'Adaboost': AdaBoostRegressor(), 
            'Gradient Boosting Regressor' : GradientBoostingRegressor()
            }

            model_report:dict() = evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            best_model = max(model_report, key = lambda k: (model_report[k][1], model_report[k][0]))
            best_model_param = model_report[best_model]

            if best_model_param[1]<0.7:
                raise CustomException("No best model Found")
            
            logging.info(f"Best model spotted: {best_model}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_filepath,
                obj=best_model
            )

            predicted = models[best_model].predict(X_test)
            r2Score = r2_score(predicted, y_test)

            return f"{best_model}: {r2Score}"
        except Exception as e:
            raise CustomException(e,sys)
