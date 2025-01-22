import sys
import os

from src.logger import logging
from src.exception import CustonmException
from src.utill import save_object,evaluate_models

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

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
        "AdaBoostRegressor" : AdaBoostClassifier(),
        "GradientBoostingRegressor" : GradientBoostingClassifier(),
        "SupportVectorClassifier" : SVC(),
        "MultinomialNB" : MultinomialNB(),
        "MLPClassifier" : MLPClassifier()
            }

            params = {
               "DecisionTreee" : {
                'criterion' : ["gini", "entropy", "log_loss"],
                'splitter' : ["best", "random"],
                'min_samples_split' : [2,3,4,5,6],
                'min_samples_leaf' : [1,2,3,4,5],
                'max_features' : ["sqrt","log2"]
                },
                "KNeighbour" : {
                    'n_neighbors' : [3,5,7,9,11],
                    'weights' : ["uniform","distance"],
                    'algorithm' : ["auto","ball_tree","kd_tree","brute"],
                    'leaf_size' : [20,25,30,35,40],
                    'p' : [1,2],
                    
                },
                "RandomForest" : {
                    'n_estimators' : [8,16,32,64,128,256],
                    'criterion' : ["gini", "entropy", "log_loss"],
                    'min_samples_split' : [2,3,4,5,6],
                    'min_samples_leaf' : [1,2,3,4,5],
                    'max_features' : ["sqrt","log2"]
                },
                "AdaBoostClassifier" : {
                                'learning_rate' : [.1,.01,0.05,0.001],
                                'n_estimators' : [8,16,32,64,128,256]
                },
                "GradientBoostingClassifier" : {
                    'loss' : ["log_loss","exponential"],
                    'subsample' : [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators' : [8,16,32,64,128,256],
                    'learning_rate' : [.1,.01,0.05,0.001],
                    'criterion' : ["friedman_mse", "squared_error"],
                    'min_samples_split' : [2,3,4,5,6],
                    'min_samples_leaf' : [1,2,3,4,5]
                },
                "SupportVectorClassifier" : {
                    'c' : [1.0,1.5,2.0,2.5],
                    'kernel' : ["linear", "poly", "rbf", "sigmoid", "precomputed"],
                    'gamma' : ["scale", "auto"],
                    'decision_function_shape' : ["ovo","ovr"]        
                },
                "MultinomialNB" : {
                    'alpha' : [1.0,1.25,1.5,1.75,2.0,3.0,4.0],    
                },
                "MLPClassifier" : {
                    'hidden_layer_sizes' : [50,100,150,200],
                    'activation' : ["identity", "logistic", "tanh", "relu"],
                    'solver' : ["lbfgs", "sgd", "adam"],
                    'learning_rate' : ["constant", "invscaling", "adaptive"],
                    'learning_rate' : [.1,.01,0.05,0.001],
                    'power_t' : [0.25,0.5,0.75,1.0],
                    'max_iter' : [100,150,200,250,300],
                    'momentum' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95],
                    'validation_fraction' : [0.1,0.2,0.3,0.4,0.5],
            
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