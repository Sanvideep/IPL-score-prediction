# IPL Score Prediction

This repository contains a Jupyter notebook that builds a machine learning model to predict the final first-innings score of an IPL match based on the current state of the innings.

The goal is to estimate the likely total by the end of 20 overs using information such as the batting team, bowling team, overs completed, current runs, wickets lost and performance in the previous overs.

---

## Objectives

This project demonstrates the complete workflow for score prediction, including:

- Loading and preparing IPL ball-by-ball data.
- Engineering features that describe the match state at a given moment.
- Converting team names to numerical values using one-hot encoding.
- Training a regression model to estimate the final score.
- Evaluating the model using standard error metrics.
- Creating a reusable function to make score predictions for any given match situation.

---

## Project Structure


All development is contained within this notebook.

---

## Data and Feature Engineering

The notebook works with a cleaned IPL dataset and converts it into a format where each row represents an innings state at a specific over.

Key features used for prediction include:

- `bat_team` – current batting team  
- `bowl_team` – current bowling team  
- `overs` – completed overs (e.g., 10.5 means 10 overs and 3 balls)  
- `runs` – total runs scored so far  
- `wickets` – total number of wickets lost  
- `runs_in_prev_5` – number of runs scored in the last 5 overs  
- `wickets_in_prev_5` – wickets lost in the last 5 overs  

### Removing the First 5 Overs

Rows where `overs < 5.0` are removed. The first 5 overs often behave very differently due to powerplay restrictions and higher volatility in scoring. Removing them allows the model to learn more stable scoring patterns for the later phases of the innings.

### Encoding Categorical Features

Team names are converted to numeric dummy variables using one-hot encoding:

```python
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
