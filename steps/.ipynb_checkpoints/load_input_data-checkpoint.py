from zenml import step
import logging
import pandas as pd

@step
def load_input(data_path: str) -> pd.DataFrame:
    '''
    Load inference data from a CSV file.

    Args:
        data_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    '''
    try:
        logging.info("Loading inference data from: %s", data_path)
        df = pd.read_csv(data_path)
        logging.info("Data loaded successfully with shape: %s", df.shape)
        return df
    except Exception as e:
        logging.exception(f"An error occurred while loading data: {e}")
        raise e
