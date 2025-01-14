import sys
import pandas as pd

from src.logger import logging
from src.exception import CustonmException
from src.utill import load_object



class predictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data = preprocessor.transform(features)
            pred= model.predict(data)
            return pred[0]

        except Exception as e:
            raise CustonmException(e,sys)

class getData:
    def __init__(self): ##Add features with self
        self.data: dict={}

    def perpare(self):
        return pd.DataFrame(self.data)