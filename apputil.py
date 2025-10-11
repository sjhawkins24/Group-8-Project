import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression

#Start by importing the merged data 
teams = pd.read_csv("CollegeFootballRankings2025.csv")

#Model building section
data = pd.read_csv("CleanedTrainingData.csv")
y = data["rank_change"]
X = data[["win_loss", "ranked_opponent", "point_differential"]]
model = LinearRegression()
model.fit(X, y)

#Function to create the data that will be used to predict
def get_pred_data(team, 
                  selected_opponent, 
                  selected_points_scored, 
                  selected_points_allowed, 
                  win, 
                  teams):
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
    final_pred_data = pred_data["win_loss", "ranked_opponent", "point_differential"]
    return(final_pred_data)

def get_pred(data, model):
    y_pred = model.predict(data)
    return(y_pred)

def get_results_text(model, 
                     team, 
                  selected_opponent, 
                  selected_points_scored, 
                  selected_points_allowed, 
                  win): 
    data = get_pred_data()
    rank_change = get_pred(data, model)
    if(win == "W"): 
        text_result = "beat"
    else: 
        text_result = "lose to"   

    if(rank_change > 0): 
        direction = "move down by"
    else: 
        direction = "move up by"         
    return_text = f"If the {team} \
        {text_result} \
            {selected_opponent} by \
                {abs(selected_points_scored - selected_points_allowed)}\
                    points, they will {direction}\
                        {abs(rank_change)}\
                            ranking points"    
    return(return_text)