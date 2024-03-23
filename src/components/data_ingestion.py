import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


"""Data ingestion configuration path"""
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts", "train.csv")
    test_data_path: str=os.path.join("artifacts", "test.csv")
    raw_data_path: str=os.path.join("artifacts", "data.csv")

"""Class for Data Ingestion function"""    
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()
    
    # Function for Initiating the data ingestion from the input data;    
    def initiate_data_ingestion(self,csv_file):
        logging.info("Data Ingestion Started")
        
        try:
            
            # Read the CSV data to Pandas Dataframe; 
            df = pd.read_csv(csv_file)
            
            # Add Log;
            logging.info(f"{csv_file} Data is read.")
            
            # Make the Train Data file;
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            
            # Save the input data to Train Data File;
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Add Log;
            logging.info(f"{csv_file} Raw Input Data is saved to Train data file - {self.ingestion_config.train_data_path}.")
            
            # Add Log;
            logging.info(f"Train Test Split intiated")
            
            # Split Train and Test Data split using 'sklearn' Library;
            train_set_data,test_set_data = train_test_split(df,random_state=42)
            
            # Save the Train data to Train CSV;
            train_set_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Save the Test data to Train CSV;
            test_set_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            # Add Log;
            logging.info(f"Train Test Split Completed and saved the data to files")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion("notebook\data\stud.csv")
    

