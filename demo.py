import numpy as np 
import pandas as pd
from regression_models import get_lr_coef, get_val_tuned_rfe
from sklearn import datasets
from sklearn.model_selection import train_test_split
from explanations import 

diabetes = datasets.load_diabetes(as_frame = True)
df = diabetes["frame"]
regressors = df.columns[0:-1]
train_df, val_df = train_test_split(df, test_size = 0.2, random_state = 123)

# print(get_lr_coef(regressors, "target", train_df, val_df))
print(get_val_tuned_rfe(train_df, val_df, regressors, "target"))