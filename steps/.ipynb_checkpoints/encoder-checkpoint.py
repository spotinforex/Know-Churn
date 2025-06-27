import logging 
from sklearn.preprocessing import OrdinalEncoder 
import pandas as pd


class Encoder():
    '''
    Encode Categorical data 
    '''
    def __init__(self, data:pd.DataFrame):
        
        self.data = data 

    def fit_transform(self):
        try:
            logging.info('Encoding data started')
            self.cat = self.data.select_dtypes(include = 'object').columns.tolist()
            self.encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = -1)
            self.data[self.cat] = self.encoder.fit_transform(self.data[self.cat])
            logging.info(" Encoding data Complete")
            return self.data

        except Exception as e:
            logging.error(f" Error encoding data {e}")
            raise e

    def transform(self):
        try:
            logging.info(" Encoding test data")

            return self.data[self.cat] = self.encoder.transform(self.data[self.cat])
        
        except Exception as e:
            logging.error(f" Error in transforming data {e}")
            raise e
            
        
        