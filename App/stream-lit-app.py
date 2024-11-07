# stream-lit-app.py

import streamlit as st
import pandas as pd
import numpy as np

from preprocessings import load_data
from preprocessings import clean_data
from preprocessings import calculate_stats
from preprocessings import preprocess_new_data
from preprocessings import load_model
from preprocessings import load_scaler
from preprocessings import load_label_encoders

def main():
    st.set_page_config(
        page_title="Valorant Performance Predictor",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load and preprocess data
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    stats = calculate_stats(cleaned_data)  # Updated to use calculate_stats
    model = load_model()
    scaler = load_scaler()
    label_encoders = load_label_encoders()
    
    st.header("Valorant Performance Predictor")
    
    # Sidebar for numerical inputs using sliders
    with st.sidebar:
        st.header("Input Parameters")
        
        # Helper function to create sliders with dynamic min, max, and value
        def create_slider(label, column):
            col_type = stats[column]['type']
            if col_type == 'int':
                return st.slider(
                    label,
                    min_value=stats[column]['min'],
                    max_value=stats[column]['max'],
                    value=stats[column]['mean']
                )
            elif col_type == 'float':
                return st.slider(
                    label,
                    min_value=stats[column]['min'],
                    max_value=stats[column]['max'],
                    value=stats[column]['mean'],
                    step=0.1
                )
        
        # Game Details
        st.subheader("Game Details")
        damage_round = create_slider("Damage per Round", 'damage_round')
        headshots = create_slider("Headshots", 'headshots')
        headshot_percent = create_slider("Headshot Percent", 'headshot_percent')
        aces = create_slider("Aces", 'aces')
        clutches = create_slider("Clutches", 'clutches')
        flawless = create_slider("Flawless Rounds", 'flawless')
        first_bloods = create_slider("First Bloods", 'first_bloods')
        kills_round = create_slider("Kills per Round", 'kills_round')
        most_kills = create_slider("Most Kills in a Game", 'most_kills')
        score_round = create_slider("Score per Round", 'score_round')
        wins = create_slider("Wins", 'wins')
        
        # Gun-1 Details
        st.subheader("Gun-1 Details")
        gun1_head = create_slider("Gun 1 Headshots", 'gun1_head')
        gun1_body = create_slider("Gun 1 Body Shots", 'gun1_body')
        gun1_legs = create_slider("Gun 1 Leg Shots", 'gun1_legs')
        gun1_kills = create_slider("Gun 1 Kills", 'gun1_kills')
        
        # Gun-2 Details
        st.subheader("Gun-2 Details")
        gun2_head = create_slider("Gun 2 Headshots", 'gun2_head')
        gun2_body = create_slider("Gun 2 Body Shots", 'gun2_body')
        gun2_legs = create_slider("Gun 2 Leg Shots", 'gun2_legs')
        gun2_kills = create_slider("Gun 2 Kills", 'gun2_kills')
        
        # Gun-3 Details
        st.subheader("Gun-3 Details")
        gun3_head = create_slider("Gun 3 Headshots", 'gun3_head')
        gun3_body = create_slider("Gun 3 Body Shots", 'gun3_body')
        gun3_legs = create_slider("Gun 3 Leg Shots", 'gun3_legs')
        gun3_kills = create_slider("Gun 3 Kills", 'gun3_kills')
        
        # Game KDA Details
        st.subheader("Game KDA Details")
        kills = create_slider("Kills", 'kills')
        deaths = create_slider("Deaths", 'deaths')
        assists = create_slider("Assists", 'assists')
    
    # Input and Prediction columns
    input_column, prediction_column = st.columns([3, 1])

    with input_column:
        st.subheader("Additional Inputs")
        # Rating selection
        rating = st.selectbox(
            "Select Rating",
            [
                'Radiant', 'Immortal 3', 'Immortal 2', 'Immortal 1', 'Diamond 3',
                'Diamond 2', 'Diamond 1', 'Platinum 3', 'Platinum 2', 'Platinum 1',
                'Gold 3', 'Gold 2', 'Gold 1', 'Silver 3', 'Silver 2','Silver 1',
                'Bronze 3', 'Unrated'  
            ]
        )

        # Agents selection
        agents = [
                'Phoenix', 'Jett', 'Reyna', 'Raze', 'Yoru', 'Neon',  # Duelists
                'Brimstone', 'Omen', 'Viper', 'Astra',               # Controllers
                'Sage', 'Cypher', 'Chamber',                         # Sentinels
                'Sova', 'Breach', 'KAY/O', 'Skye'                    # Initiators
                ]
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Agent 1")
            agent_1 = st.selectbox("Select Agent 1", agents, key='agent1')
            st.subheader("Agent 2")
            agent_2 = st.selectbox("Select Agent 2", agents, key='agent2')
            st.subheader("Agent 3")
            agent_3 = st.selectbox("Select Agent 3", agents, key='agent3')
        
        with col2:
            # Guns selection
            guns = ['Classic', 'Shorty', 'Frenzy', 'Ghost', 'Sheriff', 'Spectre', 'Bucky',
                    'Judge', 'Bulldog',  'Guardian', 'Phantom', 'Vandal', 'Marshal', 'Operator', 
                    'Ares', 'Odin']
            st.subheader("Gun 1")
            gun1_name = st.selectbox("Select Gun 1", guns, key='gun1_name')
            st.subheader("Gun 2")
            gun2_name = st.selectbox("Select Gun 2", guns, key='gun2_name')
            st.subheader("Gun 3")
            gun3_name = st.selectbox("Select Gun 3", guns, key='gun3_name')

    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'rating': [rating],
        'damage_round': [damage_round],
        'headshots': [headshots],
        'headshot_percent': [headshot_percent],
        'aces': [aces],
        'clutches': [clutches],
        'flawless': [flawless],
        'first_bloods': [first_bloods],
        'kills_round': [kills_round],
        'most_kills': [most_kills],
        'score_round': [score_round],
        'wins': [wins],
        'agent_1': [agent_1],
        'agent_2': [agent_2],
        'agent_3': [agent_3],
        'gun1_name': [gun1_name],
        'gun1_head': [gun1_head],
        'gun1_body': [gun1_body],
        'gun1_legs': [gun1_legs],
        'gun1_kills': [gun1_kills],
        'gun2_name': [gun2_name],
        'gun2_head': [gun2_head],
        'gun2_body': [gun2_body],
        'gun2_legs': [gun2_legs],
        'gun2_kills': [gun2_kills],
        'gun3_name': [gun3_name],
        'gun3_head': [gun3_head],
        'gun3_body': [gun3_body],
        'gun3_legs': [gun3_legs],
        'gun3_kills': [gun3_kills],
        'kills': [kills],
        'deaths': [deaths],
        'assists': [assists],
    })

    # Initialize prediction to none
    prediction = None

    # Prediction
    with input_column:
        if st.button("Predict"):
            new_data = input_data
            processed_data = preprocess_new_data(new_data, label_encoders, scaler)
            prediction = model.predict(processed_data)
            prediction = np.clip(prediction, 0, 100)
    
    with prediction_column:
        prediction_container = st.container()
        with prediction_container:
            st.header("Prediction")
            if prediction is not None:
                win_percent = prediction[0]
                if win_percent <= 66:
                    st.error(f"Predicted Win Percentage: {win_percent:.2f}%")
                else:
                    st.success(f"Predicted Win Percentage: {win_percent:.2f}%")

                # Display motivational message based on prediction
                if win_percent < 40:
                    st.error("Better luck next time! Give it your all and enjoy the game.")
                elif 40 <= win_percent <= 65:
                    st.warning("Going well! Keep pushing hard you can do this. Dominate them!!")
                elif 66 <= win_percent <= 90:
                    st.success("Well done! Just make sure you win and the game doesn't switch at the end.")
                else:
                    st.success("Fantastic! You're on the verge of winning!")
            else:
                st.info("No prediction made yet. Click 'Predict' to generate Performance Prediction.")

if __name__ == "__main__":
    main()
