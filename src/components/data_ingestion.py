import os 
import sys 
from src.exception import CustomException
from src.logger import logging


import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass()
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion method starts")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset read as pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")
            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)
        
        except Exception as e:
            raise CustomException(e, sys) from e




if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
# In this code, we have defined a DataIngestion class that handles the data ingestion process.
# The class has a method called initiate_data_ingestion that reads a CSV file, splits the data into training and testing sets, and saves them to specified paths.
# The paths are defined in a DataIngestionConfig dataclass. The code also includes error handling using a custom exception class and logging for better traceability.
# The main block at the end allows the script to be run directly, which will execute the data ingestion process.
# The code is structured to be modular and reusable, making it easier to maintain and extend in the future.
# The use of dataclasses for configuration helps in managing the parameters cleanly.
# The logging statements provide insights into the flow of the program, which is useful for debugging and monitoring.
# Overall, this code serves as a foundational component for a data pipeline, specifically focusing on the ingestion phase.
# It can be integrated with other components like data transformation and model training in a complete machine learning workflow.
# The code is designed to be part of a larger machine learning project, where data ingestion is a crucial step in preparing the data for analysis and modeling.
# The use of train-test split ensures that the model can be evaluated on unseen data, which is essential for assessing its performance and generalization capabilities.
# The code is also structured to be easily extendable, allowing for future enhancements such as adding more preprocessing steps or integrating with different data sources.
# The logging and exception handling mechanisms are in place to ensure that any issues encountered during the data ingestion process are properly reported and can be addressed promptly.