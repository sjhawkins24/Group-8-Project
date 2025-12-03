# CFB-Predictor

## Abstract

Every week since 1936, the Associated Press surveys members of the media covering college football to rank the different Football Bowl Subdivision teams, generating the AP Top 25. This is the defacto official ranking for college teams and becomes an important piece of information the College Football Playoff committee takes into account when determining which teams will get the chance to play for the National Football Championship.

While the AP poll is the gold standard ranking system, it is simply the aggregation of opinions, and often the topic of debate on the Monday morning sports shows. Does Ohio State's win over Texas justify their spot at #1 when Indiana beat Oregon in Eugene? Early on in the season, did Notre Dame deserve to be ranked when they started 0-2? IF Vanderbilt would have beaten Texas, would their chances of a playoff berth be more certain?

The goal of this app is to allow the user to run different hypothetical match-ups and see what the difference in ranking would be. It is an imperfect tool, as the rankings are still subjective, but it does allow for a little peak behind the curtain.

## Data Description

The data used was scraped from ESPN and provides the weekly schedules for teams as well as their AP rankings. There are two main data sets we have been working with. The first, used for training the model contains the weekly details for each FBS team from 2021-2024. Here we have for each record, a team, their opponent, game details (such as which team won, was it a home game, did it go to overtime etc) and the AP Rank assigned to the team after a given game was played.  The app runs on this same data, but for the 2025 season. This provides the schedules for each team and the current rank for a specific team. 

## Algorithm Description

This project uses a **hybrid ensemble approach** combining regression and classification models to predict ranking movements:

- **Regression Model**: Predicts continuous rank change (e.g., -2.5, +1.3) using historical game performance
- **Classification Model**: Predicts direction of movement (up/down/flat) with confidence probabilities
- **Feature Engineering**: Incorporates team Elo ratings (FiveThirtyEight-style), game margins, win streaks, and opponent strength
- **Temporal Validation**: Time-series cross-validation ensures models respect chronological order of games
- **Performance**: Achieves ~2.4 MAE on rank change prediction with 63.8% direction accuracy

The Elo rating system (K=22.0, HFA=55 points) evaluates team strength from game outcomes across the season, while the hybrid models learn how rankings respond to wins, losses, and opponent caliber.

## Tools Used

Beyond the algorithm chosen, we used several tools to build our app. We host the app data in Google drive and use `streamlit_gsheets` to prevent the app from having to load the data from github every time it boots up. We use web-scraping to create the datasets used. We also use the test/train split to ensure that our models are not over-fitting.

(PLEASE CHECK)

## Ethical Concerns

When we initially started this project, we had scraped the 2020 season data as well. However, upon digging in, we found that the pandemic really distorted that data. Not all teams played the same number of games, star players were often out, and it was such a strange time to be in college. We decided that because of this distorted data, if we included it in the models, it could create bias problems in the model.

The other potential data issue comes with the NCAA allowing for players to get paid for their name and likeness (NIL) ahead of the 2021 season. This was initially designed to allow players to be paid when they appeared in video games like EA Sports or receive royalties from their jersey sales. As time has gone on, many schools have an NIL budget so they can pay players directly to come to their school. This means the schools with the biggest budgets can pay for the best players. This may bias some of the data we have collected, where the dynasty schools (Ohio State, Alabama etc), may show less movement in the 2024 season than in the 2021 season. 

