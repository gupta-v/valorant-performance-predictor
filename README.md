# Valorant Performance Prediction

This project, **Valorant Performance Prediction**, is a machine learning application designed to predict player performance in Valorant based on a variety of in-game statistics. The prediction is powered by a Gradient Boosting Regressor model trained using scikit-learn.

## Table of Contents

- [Project Overview](#project-overview)
- [Uses and Scope](#uses-and-scope)
- [File Structure](#file-structure)
- [Software and Tools Requirements](#software-and-tools-requirements)
- [Getting Started](#getting-started)
- [Data Description](#data-description)
- [Usage](#usage)
- [Project Steps](#project-steps)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Data Preprocessing](#3-data-preprocessing)
  - [4. Model Training](#4-model-training)
  - [5. Model Evaluation](#model-evaluation)
- [Future Enhancements](#future-enhancements)
- [Hosted Link](#hosted-link)
- [Acknowledgments](#acknowledgments)

## Project Overview

Valorant Performance Prediction aims to analyze and model player statistics to predict performance metric win percentage. The project follows a machine learning pipeline, including data ingestion, cleaning, preprocessing, and model training.

## Uses and Scope

The _Valorant Performance Prediction_ project is a powerful tool for Valorant players, offering insights and analysis that can enhance strategic decision-making and gameplay:

- **Performance Analysis**: Players can input various game statistics and scenarios to analyze potential outcomes, understanding how different factors impact their performance.
- **Win Probability Estimation**: By providing a predicted win percentage, users can make data-driven decisions about their gameplay, such as agent selection and weapon usage.
- **Scenario Simulation**: The model allows for the simulation of different in-game situations, helping players experiment with strategies and observe their effects on win probability.
- **Community Engagement**: Content creators and analysts can use the project to engage the gaming community with data-driven content like match predictions and strategic breakdowns.
- **Future Extensions**: The project can be extended to include features like real-time data analysis, integration with gaming APIs, or expanded predictive capabilities.

## File Structure

```plaintext
valorant-performance-predict
│
├── App
│   ├── stream-lit-app.py       # Streamlit application script
│   └── preprocessings.py       # Preprocessing functions used in Streamlit App
│
├── data
│   └── val_stats.csv           # CSV data file with player statistics
│
├── steps
│   ├── data_ingestion.py       # Script for loading data
│   ├── data_cleaning.py        # Script for cleaning data
│   ├── data_preprocessing.py   # Script for data preprocessing
│   └── model_training.py       # Script for training and evaluating the model
│
├── label_encoders.pkl          # Pickle file for saved label encoders
├── model.pkl                   # Pickle file for the trained model
├── scaler.pkl                  # Pickle file for the scaler
└── README.md                   # Project documentation
```

## Software and Tools Requirements

1. [GitHub Account](https://github.com/)
2. [Streamlit Account](https://streamlit.io/)
3. [Python 3](https://www.python.org/downloads/)
4. [VSCode IDE](https://code.visualstudio.com/)
5. [Git CLI](https://git-scm.com/book/en/v2/Getting-Started-The-Command-Line)

## Getting Started

### Prerequisites

- **Python**: Version 3.7 or higher
- **pip**: Ensure `pip` is installed for managing project dependencies

### Installation

- Clone the repository:

  ```sh
  git clone https://github.com/gupta-v/valorant-performance-predict.git
  ```

- Navigate to the project directory:

  ```sh
  cd valorant-performance-predict
  ```

- Install the required packages using requirements.txt:

  ```sh
  pip install -r requirements.txt
  ```

## Data Description

The data used in this project (val_stats.csv) includes various player statistics, such as:

- headshots: Number of headshots
- kills: Number of kills
- deaths: Number of deaths
- assists: Number of assists
- agent_1, agent_2, agent_3: Agents played
- gun1_name, gun1_head, gun1_body: Gun Details
- Additional columns representing performance metrics and weapon usage

## Usage

### Running the Model Training Pipeline

- Ensure all dependencies are installed.
- Run train model step of the pipeline using the script provided in the steps folder.
  ```sh
  python steps/model_training.py
  ```
- The model will be trained and saved in the model.pkl file.
- The scaler and label encoders will be saved in the scaler.pkl and label_encoders.pkl file.

### Running the Streamlit Application

- Navigate to the App directory:
  ```sh
  cd App
  ```
- Run the Streamlit application using the command:
  ```sh
  streamlit run stream-lit-app.py
  ```
- The application will be hosted on local server.
  - Open your browser and go to http://localhost:8501 to use the web interface.

## Project Steps

### 1. Data Ingestion:

- Script: data_ingestion.py
- Description: Loads the player statistics data from data/val_stats.csv using pandas and logs the data loading process.

### 2. Data Cleaning:

- Script: data_cleaning.py
- Description: Cleans the data by dropping unnecessary columns, filling missing values, and converting data types for numerical processing.

### 3. Data Preprocessing:

- Script: datapreprocessing.py
- Description: Processes the data by calculating performance ratios, encoding categorical variables, and scaling numerical features. The data is split into training and testing sets.

### 4. Model Training:

- Script: model_training.py
- Description: Trains a Gradient Boosting Regressor on the preprocessed data, evaluates its performance using RMSE and R² score, and saves the model, scaler, and encoders.

### Model Evaluation

- Metrics Used:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- The model's performance is evaluated on the test set, and the results are printed to the console.

## Future Enhancements

- Incorporate additional features to improve prediction accuracy.
- Experiment with different regression models and hyperparameters.
- Implement cross-validation for better model evaluation.
- Develop a more comprehensive front-end interface.

## Hosted Link

- The project is hosted on Streamlit Cloud and can be accessed at: https://valorant-performance-prediction-gupta-v.streamlit.app

## Acknowledgments

- Inspired by the exciting gameplay and competitive nature of Valorant.
- Thanks to the open-source community for the tools and libraries used.
