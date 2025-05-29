import pandas as pd
import numpy as np

concept_data = pd.read_csv("data/concepts.csv")
train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print(train_data.shape)
print(train_data.isnull().sum())
