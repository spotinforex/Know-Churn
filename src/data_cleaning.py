from typing import Union 
import logging 
from abc import ABC, abstractmethod
import pandas as pd 
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    '''
    Abstract class defining strategy for handling data 
    '''
    @abstractmethod 
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, tuple]:
        pass

class DataPreProcessStrategy(DataStrategy):
    '''
    Strategy for preprocessing data based on our prototype
    check the notebook for references
    '''
    def handle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info('Preprocessing has started')
                         
            # Dropping unused columns
            data = df.drop(['PromptInput', 'CustomerFeedback', 'customerID'], axis=1)
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
            data['Churn'] = data['Churn'].map({'No': 0, 'Yes': 1})
            
            logging.info('Preprocessing complete')
            return data 
        except Exception as e:
            logging.error(f'Error in preprocessing: {e}')
            raise e

class DataDivideStrategy(DataStrategy):
    '''
    Strategy for splitting the data
    '''
    def handle_data(self, df: pd.DataFrame) -> tuple:
        try:
            logging.info('Data division has started')
            
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True
            )

            logging.info('Data division complete')
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f'Error in dividing: {e}')
            raise e

class DataHandler:
    '''
    Class to activate the preprocessing and splitting function
    '''
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, tuple]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(e)
            raise e


        