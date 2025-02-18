import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from scipy.stats import zscore
import seaborn as sns


from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, TargetEncoder, FunctionTransformer
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
train_pct = 1.0
MIN_FREQ_CAT = 1000  

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


# Confirming training values and labels match, 
n_train_samples = len(train_values.index) 
n_train_labels = len(train_labels.index)
if n_train_samples != n_train_labels: 
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



# 2. Data Preprocessing ----------------------------------------------------

# Converting 'date_recorded' into a numerical feature: 
#train_values["date_recorded"] = pd.to_datetime(train_values.date_recorded, format="%Y-%m-%d") 

# Creating transformer for datetime 
def transform_date_sin_cos(df): 
    df["date_recorded"] = pd.to_datetime(df.date_recorded, format="%Y-%m-%d") 
    df["day"] = df["date_recorded"].dt.day
    df["month"] = df["date_recorded"].dt.month
    df["year"] = df["date_recorded"].dt.year 

    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 31)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 31)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["year_sin"] = np.sin(2 * np.pi * (df["year"] % 10) / 3)  
    df["year_cos"] = np.cos(2 * np.pi * (df["year"] % 10) / 3)

    df.drop(columns=["day", "month", "year", "date_recorded"], inplace=True)
    return df

date_transformer = FunctionTransformer(transform_date_sin_cos, validate=False)

numeric_cols = train_values.select_dtypes(include=["int64", "float64"], exclude=["object", "datetime"]).drop(columns=["id"]).columns
categoric_cols = train_values.select_dtypes(include=["object"], exclude=["int64", "float64", "datetime"]).columns


if categorical_preprocessing == "OneHotEncoder":
   encoder = OneHotEncoder(
       min_frequency= MIN_FREQ_CAT
       , handle_unknown='infrequent_if_exist'
   , sparse_output= False 
   )   
elif categorical_preprocessing == "OrdinalEncoder":
   encoder = OrdinalEncoder(
      handle_unknown="use_encoded_value"
      , unknown_value=-1
      , encoded_missing_value= -1 
      , min_frequency = MIN_FREQ_CAT
   )
elif categorical_preprocessing == "TargetEncoder":
   encoder = TargetEncoder()


# Numerical preprocessing
if numerical_preprocessing == "StandardScaler" :
   scaler = StandardScaler()
else:
   scaler = "passthrough"
   

# Transformer object with scaler and encoder
preprocessor = ColumnTransformer(
   transformers = [
        ('date', date_transformer,["date_recorded"] ),
        ('num', scaler, numeric_cols),
        ('cat', encoder, categoric_cols)],
   verbose = False)


# Split the data into train and test sets 
# We are only interested in testing accuracy so there will be no validation set
X_train = train_values.drop(columns=["id"])
y_train = train_labels.drop(columns=["id"])
# X_train, X_val, y_train, y_val = train_test_split(train_values, train_labels, train_size = train_pct)

# Apply to the training data 
if categorical_preprocessing == "TargetEncoder": 
    # Target Encoder requires labels during fitting
    X_train_transformed = preprocessor.fit_transform(X_train, y_train["status_group"])
else: 
    X_train_transformed = preprocessor.fit_transform(X_train)
   

# 3. Model Training and Evaluation -------------------------------------------------------
if model_type == "LogisticRegression": 
    model = LogisticRegression()
elif model_type == "RandomForestClassifier": 
    model = RandomForestClassifier()
elif model_type == "GradientBoostingClassifier": 
    model = GradientBoostingClassifier( )
elif model_type == "HistGradientBoostingClassifier":
    model = HistGradientBoostingClassifier()
elif model_type == "MLPClassifier":
    model = MLPClassifier()


model.fit(X_train_transformed, y_train.values.ravel())

# Cross Validation on the training set
folds = KFold(n_splits=5, random_state=100, shuffle=True)
cv = cross_val_score(estimator=model,
                     X=X_train_transformed,
                     y=y_train.values.ravel(),
                     cv=folds,
                     scoring='accuracy')
print("The cross validation accuracies:")
print(cv)
print("Mean of the cross validation scores is: ", cv.mean())
print("Standard dev of the cross validation scores is: ", cv.std())



# 4. Prediction Generation -------------------------------------------------
# Transform test data with same encoder
X_test = preprocessor.transform(test_values)

# Make prediction
y_test = model.predict(X_test)
output_test = pd.DataFrame({"id": test_values["id"].values, "status_group": y_test})

# Write prediction to file 
output_test.to_csv(test_output_file, index=False)
