import seaborn as sns
import pandas as pd

#Start by importing the merged data 
teams = pd.read_csv("CollegeFootballRankings2025.csv")

#Function to create the data that will be used to predict
def get_pred_data():
    #check if the opponent is ranked 
    opp = teams[pd.Series(teams["Team"]).str.contains(selected_opponent).tolist()]
    opp_rank = opp["AP/CFP"].iloc[-1] != "--"
    pred_data = teams[(teams["Team"] == team) & (teams["opponent"] == selected_opponent)]

    pred_data["points_allowed"] = selected_points_allowed
    pred_data["points_scored"] = selected_points_scored
    pred_data["point_differential"] = selected_points_scored - selected_points_allowed
    pred_data["OT"] = 0 
    pred_data["OT_num"] = 0 
    pred_data["win_loss"] = win
    if(pred_data["home_game"]):
        pred_data["home_game"] = 1
    else: 
        pred_data["home_game"] == 0 
    pred_data["ranked_opponent"] = opp_rank
    return(pred_data)