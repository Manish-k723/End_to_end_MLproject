import pandas as pd
import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class data_ingestion_config:
    train_data_path:str = os.path.join('artifacts', "train.csv")
    test_data_path:str = os.path.join('artifacts', "test.csv")
    raw_data_path:str = os.path.join('artifacts', "raw.csv")
class data_ingestion:
    def __init__(self):
        self.ingestion_config = data_ingestion_config()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion method or component initiated")
        try:
            df = pd.read_csv('Notebook\StudentsPerformance.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, header = True, index = False)

            logging.info("Train Test split initiated")

            train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)
            train_df.to_csv(self.ingestion_config.train_data_path, header = True, index = False)
            test_df.to_csv(self.ingestion_config.test_data_path, header = True, index = False)

            logging.info('Data Ingetion Completed')

            return (
                self_ingestion.train_data_path, self_ingestion.test_data_path
            )

        except Exception as e:
            raise CustomeException(e,sys)
if __name__ == "__main__":
    obj = data_ingestion()
    obj.initiate_data_ingestion()