import pandas as pd
import torch
import numpy as np
import matplotlib as plt


dataset_training = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dataset_testing = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
input_cols = dataset_training.columns[1:]
target_cols = dataset_training.columns[0]
categorical_cols = dataset_training.select_dtypes(include=object).columns.values

def data_to_numpy(df):
    dataframe = df.copy(deep=True)
    for col in categorical_cols:
        dataframe[col] = dataframe[col].astype('category').cat.codes
    input = dataframe[input_cols].to_numpy()
    target = dataframe[target_cols].to_numpy()
    return input, target


X_train, Y_train = data_to_numpy(dataset_training)
X_test, Y_test = data_to_numpy(dataset_testing)

