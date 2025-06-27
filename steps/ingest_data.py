import logging
import pandas as pd
from zenml import step
from typing import Annotated

class IngestData:
    '''
       Ingesting data from data path
    '''
    def __init__(self, data_path: str):
        '''
           Args:
                 data_file
        '''
        self.data_path = data_path
    

    def get_data(self):
        '''
           ingesting the data 
        '''
        logging.info(f" ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path: str) -> Annotated[pd.DataFrame, 'raw_data']:
     '''
     Ingesting data from the data path

     Args:
          data_path: path to the data 
     Returns:
          pd.Dataframe: the ingested data 
     '''
     try:
         ingest_data = IngestData(data_path)
         df = ingest_data.get_data()
         return df

     except Exception as e:
            logging.error(f"Error while ingesting data: {e}")
            raise e
