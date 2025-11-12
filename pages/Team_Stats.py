import streamlit as st
import pandas as pd

st.title("Team Stats Analysis")

df = pd.read_csv("mergedTrainingData.csv")

selected_team = st.selectbox("Select a team:", sorted(df["Team"].unique()))
team_data = df[df["Team"] == selected_team]

st.subheader(f"Statistics for {selected_team}")
st.dataframe(team_data)