import pickle
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    logging.info("Model loaded successfully.")

# Load the scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    logging.info("Scaler loaded successfully.")

# Load the label encoders
with open('label_encoders.pkl', 'rb') as encoders_file:
    label_encoders = pickle.load(encoders_file)
    logging.info("Label encoders loaded successfully.")


def preprocess_new_data(new_data, label_encoders, scaler):
    # Apply any feature engineering you did during training
    # Here's an example assuming you had similar features as before
    new_data['kd_ratio'] = new_data['kills'] / new_data['deaths'].replace(0, np.nan)
    new_data['kda_ratio'] = (new_data['kills'] + new_data['assists']) / new_data['deaths'].replace(0, np.nan)
    new_data['ka_ratio'] = new_data['kills'] / new_data['assists'].replace(0, np.nan)
    new_data['da_ratio'] = new_data['deaths'] / new_data['assists'].replace(0, np.nan)
    new_data.fillna(0, inplace=True)

    # Drop original columns if necessary
    new_data.drop(['kills', 'assists', 'deaths'], axis=1, inplace=True)

    # Encode categorical variables using label encoders
    for col, encoder in label_encoders.items():
        new_data[col] = encoder.transform(new_data[col])  # Handle unseen categories appropriately

    # Scale numerical features
    numerical_cols = [
        'damage_round', 'headshots', 'headshot_percent', 'aces', 'clutches',
        'flawless', 'first_bloods',  'kills_round', 'most_kills',
        'score_round', 'wins', 'gun1_head', 'gun1_body', 'gun1_legs', 'gun1_kills',
        'gun2_head', 'gun2_body', 'gun2_legs', 'gun2_kills',
        'gun3_head', 'gun3_body', 'gun3_legs', 'gun3_kills',
        'kd_ratio','kda_ratio', 'ka_ratio', 'da_ratio'
    ]
    
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])
    
    return new_data

# Sample new data for prediction
new_data = pd.DataFrame({
    # Populate this DataFrame with your input data, matching the feature names
    'rating': ['Radiant'],  # Example categorical data
    'damage_round': [135],
    'headshots': [992],
    'headshot_percent': [24.9],
    'aces': [0],
    'clutches': [140],
    'flawless': [80],
    'first_bloods': [161],
    'kills_round': [0.7],
    'most_kills': [29],
    'score_round': [208],
    'wins': [59],
    'agent_1': ['Fade'],
    'agent_2': ['Viper'],
    'agent_3': ['Omen'],
    'gun1_name': ['Vandal'],
    'gun1_head': [35],
    'gun1_body': [59],
    'gun1_legs': [5],
    'gun1_kills': [802],
    'gun2_name': ['Phantom'],
    'gun2_head': [33],
    'gun2_body': [62],
    'gun2_legs': [5],
    'gun2_kills': [220],
    'gun3_name': ['Classic'],
    'gun3_head': [36],
    'gun3_body': [60],
    'gun3_legs': [3],
    'gun3_kills': [147],
    'kills': [1506],
    'deaths': [1408],
    'assists': [703],
})

# Preprocess the new data
prepared_data = preprocess_new_data(new_data, label_encoders, scaler)

print(prepared_data)
# Make a prediction
predicted_value = model.predict(prepared_data)
print(predicted_value)
# Clip the predicted value to be within 0 and 100


# print(model.feature_names_in_)
