import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import numpy as np
from data_ingestion import load_data
from data_cleaning import clean_data

from collections import namedtuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and clean the dataset
dataset = load_data()
data = clean_data(dataset)

# Define a namedtuple for structured returns
PreprocessedData = namedtuple('PreprocessedData', ['X_train', 'X_test', 'y_train', 'y_test', 'scaler', 'label_encoders'])

def preprocess_data(data):
    try:
        logging.info("Data Preprocessing Started")
        ## Feature Engineering
        # Calculate K/D Ratio (avoiding division by zero)
        data['kd_ratio'] = np.where(data['deaths'] != 0, data['kills'] / data['deaths'], 0)

        # Calculate KDA Ratio (avoiding division by zero)
        data['kda_ratio'] = np.where(data['deaths'] != 0, (data['kills'] + data['assists']) / data['deaths'], 0)

        # Calculate K/A Ratio (avoiding division by zero)
        data['ka_ratio'] = np.where(data['assists'] != 0, data['kills'] / data['assists'], 0)

        # Calculate D/A Ratio (avoiding division by zero)
        data['da_ratio'] = np.where(data['assists'] != 0, data['deaths'] / data['assists'], 0)

        # Replace any NaN values that resulted from division by zero with 0
        data.fillna(0, inplace=True)

        # Drop the columns to reduce redundancy 
        data.drop(['kills', 'assists', 'deaths'], axis=1, inplace=True)

        # Specify your categorical and numerical columns
        categorical_cols = ['rating', 'agent_1', 'agent_2', 'agent_3', 'gun1_name', 'gun2_name', 'gun3_name']
        numerical_cols = [
            'damage_round', 'headshots', 'headshot_percent', 'aces', 'clutches', 
            'flawless', 'first_bloods', 'kills_round', 'most_kills', 
            'score_round', 'wins', 'gun1_head', 'gun1_body', 'gun1_legs', 'gun1_kills', 
            'gun2_head', 'gun2_body', 'gun2_legs', 'gun2_kills', 
            'gun3_head', 'gun3_body', 'gun3_legs', 'gun3_kills',
            'kd_ratio', 'kda_ratio', 'ka_ratio', 'da_ratio'
        ]

        # Define your target variable and features
        X = data.drop(columns=['win_percent'])  # Drop the target variable
        y = data['win_percent']  # Your target variable

        # Step 1: Encode categorical features
        label_encoders = {}
        for col in categorical_cols:
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col])  # Fit and transform each categorical column
            label_encoders[col] = encoder  # Store the encoder for future use if needed

        # Step 2: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 3: Scale the numerical features
        scaler = StandardScaler()

        # Fit the scaler on the training data and transform both train and test data
        X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
        
        logging.info("Data Preprocessing Finished")
        
        return PreprocessedData(X_train, X_test, y_train, y_test, scaler, label_encoders)
    
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise e


