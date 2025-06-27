import logging
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


class Encoder:
    '''
    Encode categorical features using OrdinalEncoder.
    '''

    def __init__(self):
        self.cat = None
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Fitting and transforming training data")

            # Identify categorical columns during training
            self.cat = data.select_dtypes(include='object').columns.tolist()
            if not self.cat:
                logging.warning("No categorical columns found")
                return data

            data = data.copy()
            data[self.cat] = self.encoder.fit_transform(data[self.cat])
            logging.info(f"Categorical columns encoded: {self.cat}")
            return data

        except Exception as e:
            logging.exception(f"Error encoding training data: {e}")
            raise e

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Transforming test/inference data")

            if self.cat is None:
                raise ValueError("Encoder has not been fitted. 'cat' is None.")

            data = data.copy()
            data[self.cat] = self.encoder.transform(data[self.cat])
            return data

        except Exception as e:
            logging.exception(f"Error transforming data: {e}")
            raise e
