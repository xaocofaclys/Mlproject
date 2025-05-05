import os 
import sys 
import numpy as np 
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging

def save_object(obj, file_path):
    """
    This function saves the given object to the specified file path using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e