import streamlit as st
import pickle
import numpy as np
import pandas as pd
import logging

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
    

def calculate_means(data):
    # Calculate the means of numerical columns
    means = {}
    numerical_cols = [
        'damage_round', 'headshots', 'headshot_percent', 'aces', 'clutches',
        'flawless', 'first_bloods',  'kills_round', 'most_kills',
        'score_round', 'wins', 'gun1_head', 'gun1_body', 'gun1_legs', 'gun1_kills',
        'gun2_head', 'gun2_body', 'gun2_legs', 'gun2_kills',
        'gun3_head', 'gun3_body', 'gun3_legs', 'gun3_kills',
        'kills','assists','deaths'
    ]
    for col in numerical_cols:
        means[col] = data[col].mean()
    return means

def preprocess_new_data(new_data, label_encoders, scaler):
    new_data['kd_ratio'] = new_data['kills'] / new_data['deaths'].replace(0, np.nan)
    new_data['kda_ratio'] = (new_data['kills'] + new_data['assists']) / new_data['deaths'].replace(0, np.nan)
    new_data['ka_ratio'] = new_data['kills'] / new_data['assists'].replace(0, np.nan)
    new_data['da_ratio'] = new_data['deaths'] / new_data['assists'].replace(0, np.nan)
    new_data.fillna(0, inplace=True)
    new_data.drop(['kills', 'assists', 'deaths'], axis=1, inplace=True)
    
    for col, encoder in label_encoders.items():
        new_data[col] = encoder.transform(new_data[col])  # Handle unseen categories appropriately

    numerical_cols = [
        'damage_round', 'headshots', 'headshot_percent', 'aces', 'clutches',
        'flawless', 'first_bloods',  'kills_round', 'most_kills',
        'score_round', 'wins', 'gun1_head', 'gun1_body', 'gun1_legs', 'gun1_kills',
        'gun2_head', 'gun2_body', 'gun2_legs', 'gun2_kills',
        'gun3_head', 'gun3_body', 'gun3_legs', 'gun3_kills',
        'kd_ratio', 'kda_ratio', 'ka_ratio', 'da_ratio'
    ]
    
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])
    
    return new_data




@st.cache_resource
def load_model():
    with open('./model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        logging.info("Model loaded successfully.")
        return model

@st.cache_resource
def load_scaler():
    with open('./scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
        logging.info("Scaler loaded successfully.")
        return scaler

@st.cache_resource
def load_label_encoders():
    with open('./label_encoders.pkl', 'rb') as encoders_file:
        label_encoders = pickle.load(encoders_file)
        logging.info("Label encoders loaded successfully.")
        return label_encoders
