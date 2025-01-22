import sys
import os

from src.logger import logging
from src.exception import CustonmException
from src.utill import save_object,evaluate_models

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

from dataclasses import dataclass

@dataclass
class modelTrainerConfig:
    trained_model_path = os.path.join("artifacts","model.pkl")

class modelTrainer:
    def __init__(self):
        self.model_train_config = modelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            x_train, y_train, x_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models ={
                "DecisionTreee": DecisionTreeClassifier(),
                "KNeighbour" : KNeighborsClassifier(),
                "RandomForest": RandomForestClassifier(),
                "AdaBoostClassifier" : AdaBoostClassifier(),
                "GradientBoostingClassifier" : GradientBoostingClassifier()
            }

            params = {
                "LinearRegressor" : {},
                "DecisionTreee" : {
                    'criterion' : ['squared_error','friedman_mse','absolute_error','poisson']
                },
                "KNeighbour" : {
                    'n_neighbors' : [3,5,7,9,11]
                },
                "RandomForest" : {
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "AdaBoostRegressor" : {
                    'learning_rate' : [.1,.01,0.05,0.001],
                    'n_estimators' : [8,16,32,64,128,256]
                },
                "GradientBoostingRegressor" : {
                    'subsample' : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators' : [8,16,32,64,128,256],
                    'learning_rate' : [.1,.01,0.05,0.001]
                }

            }

            logging.info("model training initiated")

            model_report:dict = evaluate_models(x_train,y_train,x_test,y_test,models,params)


            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            if best_model_score < 0.6:
                raise CustonmException("No best model found")
            logging.info(f"Best model :{best_model_name} score :{best_model_score}")

            best_model = models[best_model_name]
            save_object(obj=best_model, file_path=self.model_train_config.trained_model_path)
        except Exception as e:
            raise CustonmException(e,sys)