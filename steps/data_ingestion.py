import pandas as pd
import os
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def load_data():
    file_path="./data/val_stats.csv"
    try:
        logging.info("Data Loading Started")
        data = pd.read_csv(file_path,low_memory=False)  
        logging.info("Data Loaded Successfully!")
        return data
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
        return None
    except Exception as e:
        logging.exception("An error occurred while loading the data.")
        return None

