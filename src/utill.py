import os
import sys
from src.exception import CustonmException
from src.logger import logging
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import pickle

def save_object(obj, file_path):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(file_path,"wb") as file :
            dill.dump(obj, file)
    except Exception as e:
        raise CustonmException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report ={}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = list(params.values())[i]
            
            gs = GridSearchCV(estimator=model,param_grid=param)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)

            score = r2_score(y_true=y_test,y_pred=y_pred)

            report[list(models.keys())[i]] = score

        return report
    except Exception as e:
        raise CustonmException(e,sys)


    except Exception as e:
        raise CustonmException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as f:
            return dill.load(f)
    except Exception as e:
        raise CustonmException(e,sys)