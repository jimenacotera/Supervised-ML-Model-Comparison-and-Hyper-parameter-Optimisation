import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from scipy.stats import zscore
import seaborn as sns


from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import optuna
import optuna.visualization as vis
import plotly

import sys
import os

# Hyperparameter fields 
train_pct = 0.8

# 1. INPUT PARSING -------------------------------------------------
supported_models = ["LogisticRegression", 
                    "RandomForestClassifier", 
                    "GradientBoostingClassifier", 
                    "HistGradientBoostingClassifier",
                    "MLPClassifier"]
supported_numerical = ["None", "StandardScaler"]
supported_categorical = ["OneHotEncoder", "OrdinalEncoder", "TargetEncoder"]


# Check number of arguments
# Check input and output files are .csv
# Check the preprocessing and model are supported
if len(sys.argv) != 8: 
    print("Error: Incorrect number of input parameters", file= sys.stderr)
    sys.exit(1)
elif not sys.argv[1].endswith(".csv"): 
    print("Error: <train-input-file> needs to be .csv file", file= sys.stderr)
    sys.exit(1)
elif not sys.argv[2].endswith(".csv"): 
    print("Error: <train-labels-file> needs to be .csv file", file= sys.stderr)
    sys.exit(1)
elif not sys.argv[3].endswith(".csv"): 
    print("Error: <test-input-file> needs to be .csv file", file= sys.stderr)
    sys.exit(1)
elif not sys.argv[7].endswith(".csv"): 
    print("Error: <test-prediction-output-file> needs to be .csv file", file= sys.stderr)
    sys.exit(1)
elif sys.argv[4] not in supported_numerical: 
    print("Error: specified <numerical-preprocessing> not supported", file= sys.stderr)
    sys.exit(1)
elif sys.argv[5] not in supported_categorical: 
    print("Error: specified <categorical-preprocessing> not supported", file= sys.stderr)
    sys.exit(1)
elif sys.argv[6] not in supported_models: 
    print("Error: specified <model-type> not supported", file= sys.stderr)
    sys.exit(1)


# Read input files 
# Check the files exist and can be read
def check_and_read_csv(filepath):
    # Check if file exists at specified location
    if not os.path.exists(filepath):
        print("Error: the file '" + filepath + "' does not exist", file= sys.stderr)
        sys.exit(1)
    
    # Check if file is empty
    if os.path.getsize(filepath) == 0:
        print(f"Error: The file '{filepath}' is empty.")
        sys.exit(1)

    try:
        # Read the file into a DataFrame
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

# Import data from specificed locations
train_values = check_and_read_csv(sys.argv[1])
train_labels = check_and_read_csv(sys.argv[2])
test_values = check_and_read_csv(sys.argv[3])

# User Input Fields
numerical_preprocessing = sys.argv[4]
categorical_preprocessing = sys.argv[5]
model_type = sys.argv[6]
test_output_file = sys.argv[7]


# Confirming that training values and labels match, 
n_train_samples = len(train_values.index) 
n_test_samples = len(test_values.index)
if n_train_samples != n_test_samples: 
    print("Error: number of training samples and labels not equal.")
    sys.exit(1)
# Check features exist
if len(train_values.columns) <= 1:
    print("Error: not enough features in training data.")
    sys.exit(1)
elif len(test_values.columns) <= 1: 
    print("Error: not enough features in testing data.")
    sys.exit(1)
# Confirm training and testing features are the same
if set(train_values.columns) != set(test_values.columns): 
    print("Error: training and testing features do not match.")
    sys.exit(1)




# TODO
    # Can also check if train labels exist
    # Can check if ids of training values and labels match up 




# 2. Data Preprocessing ----------------------------------------------------

# Converting 'date_recorded' into a numerical feature: 
# the number of days since the first recorded date in the dataset.
# TODO sin cos transformation
train_values["date_recorded"] = pd.to_datetime(train_values.date_recorded, format="%Y-%m-%d")
first_recorded_date = train_values["date_recorded"].min()
new_dates = train_values["date_recorded"] - first_recorded_date
n_days_since_first = [x.days for x in new_dates]
train_values["date_recorded"] = n_days_since_first


numeric_cols = train_values.select_dtypes(include=["int64", "float64"], exclude=["object"]).drop(columns=["id"]).columns
categoric_cols = train_values.select_dtypes(include=["object"], exclude=["int64", "float64"]).columns


# Outlier Handling in Numeric Fields through imputation
# Remove row where construction year is 0 -> missing data
mask = train_values['construction_year'] != 0
print(train_values.shape)
train_values_filt = train_values[mask].reset_index(drop=True)
train_labels_filt = train_labels[mask].reset_index(drop=True)
train_values = train_values_filt
train_labels = train_labels_filt

# Removing amount_tsh column from training data due to high # of NaNs
train_values.drop(columns=["amount_tsh"])