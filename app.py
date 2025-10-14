import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
# from apputil import *

# reading the csv and creating a team list
df = pd.read_csv('mergedTrainingData.csv')
teams = df['Team'].unique()

# Simple mock prediction for the time being
def mock_predict(data):
    return np.random.uniform(-5, 5)

# Setting the title of the Streamlit page
st.title('College Football Ranking Predictor')

# Setting an image
st.image('StreamlitPic.jpg', width = 1000)

# Setting the input for the playing team
st.header('Team Selection')
team = st.selectbox('Select your team:', teams)
st.write(f'Selected team: {team}')

# Extract the FPI value for the team from the 'Team' column
team_rank = df.loc[df['Team'] == team, 'FPI'].iloc[0]

# Display the team rank in Streamlit
st.subheader('Current Team Ranking')
st.write(f"The Team's Current Rank is: {team_rank}")

# Setting the input for opponent
st.header('Opponent Selection')
opponent = st.selectbox('Select an opponent:', teams)
st.write(f'Selected opponent: {opponent}')

# Extract the FPI value for the opponent from the 'Team' column
opponent_rank = df.loc[df['Team'] == opponent, 'FPI'].iloc[0]

# Display the opponent rank in Streamlit
st.subheader("Opponent's Current Ranking")
st.write(f"The Opponent's Current Rank is: {opponent_rank}")

# Selecting if it is a home game
st.header('Home or Away Game')
home_game = st.selectbox('Is this a Home Game?', ['Yes', 'No'])
hg = 'Home' if home_game == 'Yes' else 'Away'
st.write(f'This is a(n) {hg} game')

# Setting the input for game outcome
# st.header('Game Outcome')
# result = st.selectbox('Game result:', ['W', 'L'])
# st.write(f'Game result: {result}')

# Setting the inputs for points_scored and points_allowed
st.header('Points Scored and Allowed')
points_scored = st.number_input('Points Scored:', min_value = 0, step = 1, value = 0)
points_allowed = st.number_input('Points Allowed:', min_value = 0, step = 1, value = 0)

# Calculating the point differential
point_differential = points_scored - points_allowed

# Calculating the outcome based on point differential
st.subheader('Game Outcome')
game_outcome = 'Win' if point_differential > 0 else 'Loss'
st.write(game_outcome)
game_result = 'beat' if game_outcome =='Win' else 'lost to'


# Setting up the dictionary for input values
pred_data = {
    'Team': team,
    'Team Rank': team_rank,
    'Opponent': opponent,
    'Opponent Rank': opponent_rank,
    'Team Points Scored': points_scored,
    'Opponent Points Scored': points_allowed,
    'Game Outcome': game_outcome,
    'home_game': hg
}

# Converting the Prediction Data to a Dataframe
prediction_df = pd.DataFrame(pred_data.items(), columns = ['Attribute', 'Value'])

# logging the inputs to ensure accuracy
st.header('Selected Options')
st.table(prediction_df.style.hide(axis='index'))

# Predicting the rank change (mock setup)
rank_change = mock_predict(pred_data)

# Testing to ensure rank_change is producing a result
st.subheader('Predicted Rank Change')
st.write(f"Rank Change: {rank_change}")

# Setting rank change variable
movement = 'move up' if rank_change < 0 else 'move down'

# Output text in Streamlit
result_text = f'If the {team} {game_result} {opponent} by \
    {point_differential} points, they will {movement} \
    by {abs(round(rank_change))} ranking points.'

# Displaying the result in Streamlit
st.header('Results')
st.write(result_text)

# currently set for integer input
# amount = st.number_input("Exercise Input: ", 
                         # value=None, 
                         # step=1, 
                         # format="%d")

# if amount is not None:
#    st.write(f"The exercise input was {amount}.")