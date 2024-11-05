import pandas as pd
import numpy as np
import logging
from data_ingestion import load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dataset =load_data()
def clean_data(data):
    try:
        logging.info("Data Cleaning Started")
        #Drop  rows with no contribution

        data=data.drop(['region','name','tag','kd_ratio'],axis=1)

        # Fill missing values in agent_2 and agent_3 with values from agent_1
        data['agent_2'] = data['agent_2'].fillna(data['agent_1'])
        data['agent_3'] = data['agent_3'].fillna(data['agent_1'])



        # List of columns to convert into numerical
        columns_to_convert = ['headshots', 'kills', 'deaths','first_bloods', 
                            'gun1_kills', 'gun2_kills', 'gun3_kills','assists']

        # Convert the specified columns to numeric types after replacing commas
        for column in columns_to_convert:
            # Convert the column to string, replace commas, and then convert to numeric
            data[column] = pd.to_numeric(data[column].astype(str).str.replace(',', ''), errors='coerce')
        logging.info("Data Cleaned Successfully")    
        return data
    
    except Exception as e:
        raise(e)



