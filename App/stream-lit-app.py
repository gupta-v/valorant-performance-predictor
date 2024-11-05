import streamlit as st
import pandas as pd
from preprocessings import load_data
from preprocessings import clean_data
from preprocessings import calculate_means
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

    
    raw_data = load_data()
    cleaned_data = clean_data(raw_data)
    means = calculate_means(cleaned_data)
    model = load_model()
    scaler = load_scaler()
    label_encoders = load_label_encoders()
    st.header("Performance Predictor")
    # Sidebar for numerical inputs using sliders
    with st.sidebar:
        st.header("Input Parameters")
        st.subheader("Game Details")
        damage_round = st.slider("Damage per Round", min_value=0, max_value=500, value=int(means['damage_round']))
        headshots = st.slider("Headshots", min_value=0, max_value=5000, value=int(means['headshots']))
        headshot_percent = st.slider("Headshot Percent", min_value=0.0, max_value=100.0, value=float(means['headshot_percent']))
        aces = st.slider("Aces", min_value=0, max_value=100, value=int(means['aces']))
        clutches = st.slider("Clutches", min_value=0, max_value=100, value=int(means['clutches']))
        flawless = st.slider("Flawless Rounds", min_value=0, max_value=100, value=int(means['flawless']))
        first_bloods = st.slider("First Bloods", min_value=0, max_value=1000, value=int(means['first_bloods']))
        kills_round = st.slider("Kills per Round", min_value=0.0, max_value=10.0, value=float(means['kills_round']))
        most_kills = st.slider("Most Kills in a Game", min_value=0, max_value=50, value=int(means['most_kills']))
        score_round = st.slider("Score per Round", min_value=0, max_value=500, value=int(means['score_round']))
        wins = st.slider("Wins", min_value=0, max_value=1000, value=int(means['wins']))

        st.subheader("Gun-1 Details")
        gun1_head = st.sidebar.slider("Gun 1 Headshots", min_value=0, max_value=500, value=int(means['gun1_head']))
        gun1_body = st.sidebar.slider("Gun 1 Body Shots", min_value=0, max_value=500, value=int(means['gun1_body']))
        gun1_legs = st.sidebar.slider("Gun 1 Leg Shots", min_value=0, max_value=100, value=int(means['gun1_legs']))
        gun1_kills = st.sidebar.slider("Gun 1 Kills", min_value=0, max_value=1000, value=int(means['gun1_kills']))

        st.subheader("Gun-2 Details")
        gun2_head = st.sidebar.slider("Gun 2 Headshots", min_value=0, max_value=500, value=int(means['gun2_head']))
        gun2_body = st.sidebar.slider("Gun 2 Body Shots", min_value=0, max_value=500, value=int(means['gun2_body']))
        gun2_legs = st.sidebar.slider("Gun 2 Leg Shots", min_value=0, max_value=100, value=int(means['gun2_legs']))
        gun2_kills = st.sidebar.slider("Gun 2 Kills", min_value=0, max_value=1000, value=int(means['gun2_kills']))

        st.subheader("Gun-3 Details")
        gun3_head = st.sidebar.slider("Gun 3 Headshots", min_value=0, max_value=500, value=int(means['gun3_head']))
        gun3_body = st.sidebar.slider("Gun 3 Body Shots", min_value=0, max_value=500, value=int(means['gun3_body']))
        gun3_legs = st.sidebar.slider("Gun 3 Leg Shots", min_value=0, max_value=100, value=int(means['gun3_legs']))
        gun3_kills = st.sidebar.slider("Gun 3 Kills", min_value=0, max_value=1000, value=int(means['gun3_kills']))

        st.subheader("Game KDA Details")
        kills = st.sidebar.slider("Kills", min_value=0, max_value=1000, value=int(means['kills']))
        deaths = st.sidebar.slider("Deaths", min_value=0, max_value=1000, value=int(means['deaths']))
        assists = st.sidebar.slider("Assists", min_value=0, max_value=1000, value=int(means['assists']))

    # Input and Prediction columns
    input_column, prediction_column = st.columns([3, 1])

    with input_column:
        st.subheader("Rating")
        rating = st.selectbox("Select Rating", ['Radiant', 'Immortal 3', 'Immortal 2', 'Immortal 1', 'Diamond 3',
                                                'Diamond 2', 'Diamond 1', 'Platinum 3', 'Platinum 2', 'Platinum 1',
                                                'Gold 3', 'Gold 2', 'Gold 1', 'Silver 3', 'Silver 2','Silver 1',
                                                'Bronze 3', 
                                                'Unrated'  
                                                ])

        # Dropdowns for categorical variables
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Agent 1")
            agent_1 = st.selectbox("Select Agent 1", ['Fade', 'Chamber', 'Yoru', 'Jett', 'Sage', 'KAY/O', 'Sova', 'Raze',
                                                      'Omen', 'Breach', 'Reyna', 'Neon', 'Skye', 'Viper', 'Brimstone',
                                                      'Phoenix', 'Astra', 'Killjoy', 'Cypher'])
        with col1:
            st.subheader("Agent 2")
            agent_2 = st.selectbox("Select Agent 2", ['Fade', 'Chamber', 'Yoru', 'Jett', 'Sage', 'KAY/O', 'Sova', 'Raze',
                                                      'Omen', 'Breach', 'Reyna', 'Neon', 'Skye', 'Viper', 'Brimstone',
                                                      'Phoenix', 'Astra', 'Killjoy', 'Cypher'])
        with col1:
            st.subheader("Agent 3")
            agent_3 = st.selectbox("Select Agent 3", ['Fade', 'Chamber', 'Yoru', 'Jett', 'Sage', 'KAY/O', 'Sova', 'Raze',
                                                      'Omen', 'Breach', 'Reyna', 'Neon', 'Skye', 'Viper', 'Brimstone',
                                                      'Phoenix', 'Astra', 'Killjoy', 'Cypher'])

        with col2:
            st.subheader("Gun 1")
            gun1_name = st.selectbox("Select Gun 1", ['Classic', 'Shorty', 'Frenzy', 'Ghost', 'Sheriff', 'Spectre', 'Bucky',
                                                      'Judge', 'Bulldog',  'Guardian', 'Phantom', 'Vandal', 'Marshal', 'Operator', 
                                                      'Ares', 'Odin' 
                                                      ])
        with col2:
            st.subheader("Gun 2")
            gun2_name = st.selectbox("Select Gun 2", ['Classic', 'Shorty', 'Frenzy', 'Ghost', 'Sheriff', 'Spectre', 'Bucky',
                                                      'Judge', 'Bulldog',  'Guardian', 'Phantom', 'Vandal', 'Marshal', 'Operator', 
                                                      'Ares', 'Odin' 
                                                      ])
        with col2:
            st.subheader("Gun 3")
            gun3_name = st.selectbox("Select Gun 3", ['Classic', 'Shorty', 'Frenzy', 'Ghost', 'Sheriff', 'Spectre', 'Bucky',
                                                      'Judge', 'Bulldog',  'Guardian', 'Phantom', 'Vandal', 'Marshal', 'Operator', 
                                                      'Ares', 'Odin' 
                                                      ])
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
            
    with prediction_column:
        prediction_container = st.container()
        with prediction_container:
            st.header("Prediction")
            if prediction is not None:
                win_percent = prediction[0]
                if win_percent <= 66:
                    st.error(f"Predicted Win Percentage: {prediction[0]:.2f}%")
                elif win_percent > 66:
                    st.success(f"Predicted Win Percentage: {prediction[0]:.2f}%")

                # Display motivational message based on prediction
                win_percentage = prediction[0]
                if win_percentage < 40:
                    st.error("Better luck next time! Give it your all and enjoy the game.")
                elif 40 <= win_percentage <= 65:
                    st.warning("Going well! Keep pushing hard you can do this. Dominate them!!")
                elif 66 <= win_percentage <= 90:
                    st.success("Well done! Just make sure you win and the game doesn't switch at the end.")
                else:
                    st.success("Fantastic! You're on the verge of winning!")
            else:
                 st.error("No prediction made yet, to generate Performance Prediction")
                 st.error("Click 'Predict'")

if __name__ == "__main__":
    main()