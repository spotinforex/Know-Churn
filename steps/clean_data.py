import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataHandler, DataDivideStrategy, DataPreProcessStrategy
from typing import Tuple
from typing_extensions import Annotated

logging.basicConfig(level=logging.INFO)

@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test']
]:
    '''
    Cleans the data and divides into train and test sets.

    Args:
        df: Raw input data.
    
    Returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
    '''
    try:
        data_cleaning = DataHandler(data=df, strategy=DataPreProcessStrategy())
        preprocessed_data = data_cleaning.handle_data()

        data_dividing = DataHandler(data=preprocessed_data, strategy=DataDivideStrategy())
        X_train, X_test, y_train, y_test = data_dividing.handle_data()

        logging.info('Data cleaning and splitting complete.')

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise e
