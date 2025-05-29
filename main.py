import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

concept_data = pd.read_csv("data/concepts.csv")
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

#! Data Preprocessing

#! Descriptive Analysis

#! Model Training
rf_model = RandomForestRegressor(n_estimators=200, random_state=2025)

#! Model Evaluation (MSE and R2)
