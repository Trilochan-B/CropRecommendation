import sys
import os


from src.logger import logging
from src.exception import CustonmException
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utill import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join("artifacts","preprocessor.pkl")

class dataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformation_obj(self):
        try:
            num_col=["N","P","K","temperature","humidity","ph","rainfall"] ## please mentaion the numerical columns name
            #cat_col=[] ## please mentaion the categorical columns name
            num_pipeline = Pipeline(
                steps=[
                    ("impute",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            # cat_pipeline = Pipeline(
            #     steps=[
            #         ("impute",SimpleImputer(strategy="most_frequent")),
            #         ("encoding",OneHotEncoder()),
            #         ("scaler",StandardScaler())
            #     ]
            # )

            logging.info("Transformation initiated")

            preprocessor = ColumnTransformer(
                {
                    ("num_pre",num_pipeline,num_col)
                    #("cat_pre",cat_pipeline,cat_col)
                }
            )

            return preprocessor
        except Exception as e:
            raise CustonmException(e,sys)
        
    def init_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_transformation_obj()

            y_train = train_df[:,-1] ## assuming last column of the df is dependend column
            y_test = test_df[:,-1] ## assuming last column of the df is dependend column

            input_train_feature = train_df[:,:-1]
            input_test_feature = test_df[:,:-1]

            logging.info("Appling transformation on train and test data")

            input_train_trans = preprocessor_obj.fit_transform(input_train_feature)
            input_test_trans = preprocessor_obj.transform(input_test_feature)

            train_arr = np.c_(input_train_trans, np.array(y_train))
            test_arr = np.c_(input_test_trans, np.array(y_test))

            save_object(preprocessor_obj, self.data_transformation_config.preprocessor_obj_path)

            logging.info("saved preprocessing object")

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_path)


        except Exception as e:
            raise CustonmException(e,sys)


    