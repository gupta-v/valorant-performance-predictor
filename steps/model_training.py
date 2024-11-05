import  pickle  # Ensure you have pickle5 installed
import numpy as np
import pandas as pd
import logging
from data_ingestion import load_data
from data_cleaning import clean_data
from data_preprocessing import preprocess_data
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load, clean, and preprocess the data
data = load_data()
cleaned_data = clean_data(data)
preprocessed_data = preprocess_data(cleaned_data)

# Unpack the preprocessed data
X_train, X_test, y_train, y_test, scaler, label_encoders = preprocessed_data

def train_model(X, y):
    try:
        logging.info("Model Training Started")
        
        # Initialize and fit the Gradient Boosting model
        gb_model = GradientBoostingRegressor(n_estimators=250, learning_rate=0.09, max_depth=8)
        gb_model.fit(X, y)

        logging.info("Model Training Finished")
        
        return gb_model
    
    except Exception as e:
        raise(e)


# Export the model, scaler, and label encoders using pickle
def save_artifacts(model, scaler, label_encoders):
    try:
        with open('model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        logging.info("Model saved successfully.")
        
        with open('scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        logging.info("Scaler saved successfully.")
        
        with open('label_encoders.pkl', 'wb') as encoders_file:
            pickle.dump(label_encoders, encoders_file)
        logging.info("Label encoders saved successfully.")
        
    except Exception as e:
        logging.error("Error while saving artifacts: %s", e)



# Train the model
model = train_model(X_train, y_train)
y_pred =model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)  # Mean Squared Error
mse=rmse **2
r2 = r2_score(y_test, y_pred)  # R-squared score

# Output the evaluation results
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-2 Score:", r2)
# Save the trained model, scaler, and encoders
save_artifacts(model, scaler, label_encoders)


