import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
#from apputil import *

# reading the csv and creating a team list
df = pd.read_csv('mergedTrainingData.csv')
teams = df['Team'].unique()

# Simple mock prediction for the time being
def mock_predict(data):
    return np.random.uniform(-5, 5)

# Setting the title of the Streamlit page
st.title('College Football Ranking Predictor')

# Setting the input for the playing team
team = st.selectbox('Select your team:', teams)
st.write(f'Selected team: {team}')

# Setting the input for opponent
opponent = st.selectbox('Select an opponent:', teams)
st.write(f'Selected opponent: {opponent}')

# Setting the input for game outcome
result = st.selectbox('Game result:', ['W', 'L'])
st.write(f'Game result: {result}')
game_result = 'beat' if result=='W' else 'lost to'

# Setting the inputs for points_scored and points_allowed
points_scored = st.number_input('Points Scored:', min_value = 0, step = 1, value = 0)
points_allowed = st.number_input('Points Allowed:', min_value = 0, step = 1, value = 0)

# Calculating the point differential
point_differential = points_scored - points_allowed
win = 1 if result == 'W' else 0
opponent_ranked = df[df['opponent'] == opponent]['FPI'].iloc[0] != '--'

# Setting up the dictionary for input values
pred_data = {
    'team': team,
    'opponent': opponent,
    'points_scored': points_scored,
    'points_allowed': points_allowed,
    'point_differential': point_differential,
    'win_loss': win,
    'ranked_opponent': opponent_ranked,
    'home_game': 1
}

# logging the inputs to ensure accuracy
st.write('Prediction Data:', pred_data)

# Predicting the rank change (mock setup)
rank_change = mock_predict(pred_data)

# Testing to ensure rank_change is producing a result
st.write(f"Rank Change: {rank_change}")

# Setting rank change variable
movement = 'move up' if rank_change < 0 else 'move down'

# Output text in Streamlit
result_text = f'If the {team} {game_result} {opponent} by \
    {point_differential} points, they will {movement} \
    by {abs(round(rank_change))} ranking points.'

# Displaying the result in Streamlit
st.write(result_text)

# currently set for integer input
# amount = st.number_input("Exercise Input: ", 
                         # value=None, 
                         # step=1, 
                         # format="%d")

# if amount is not None:
#    st.write(f"The exercise input was {amount}.")