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
    #Issue with str contains not accepting argument 
    opp = teams[pd.Series(teams["Team"]).str.startswith(selected_opponent).tolist()]
    opp_rank = opp["AP/CFP"].iloc[-1] != "--"
    
    pred_data = teams[(teams["Team"] == team)& (teams["opponent"] == selected_opponent)]

    pred_data["points_allowed"] = selected_points_allowed
    pred_data["points_scored"] = selected_points_scored
    pred_data.loc[:,"point_differential"] = pred_data["points_scored"] -  pred_data["points_allowed"]
    pred_data.loc[:,"OT"] = 0 
    pred_data.loc[:,"OT_num"] = 0 
    #print(win)
    pred_data.loc[:,"win_loss"] = win[0]
    if(win[0] == "W"):
            pred_data.loc[:,"win_loss"] = 1
    else: 
            pred_data.loc["", "win_loss"] == 0 
    pred_data["ranked_opponent"] = opp_rank
    final_pred_data = pred_data[["win_loss", "ranked_opponent", "point_differential"]]
    return(final_pred_data)

def get_pred(data, model):
    y_pred = model.predict(data)
    return(y_pred)

def get_results_text(model, 
                     team, 
                  selected_opponent, 
                  selected_points_scored, 
                  selected_points_allowed, 
                  win, 
                  teams): 
    data = get_pred_data(team, 
                  selected_opponent, 
                  selected_points_scored, 
                  selected_points_allowed, 
                  win, 
                  teams)
    rank_change = get_pred(data, model)
    if(win == "W"): 
        text_result = "beat"
    else: 
        text_result = "lose to"   

    if(rank_change > 0): 
        direction = "move down by"
    else: 
        direction = "move up by"         
    return_text = f"If the {team} {text_result} {selected_opponent} by {abs(selected_points_scored - selected_points_allowed)} points, they will {direction} {abs(rank_change[0])} ranking points"    
    return(return_text)
