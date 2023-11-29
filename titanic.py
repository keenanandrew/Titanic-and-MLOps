# A script for the Titanic dataset
# For practicing mlflow, optuna and other MLOps libraries

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import the data

train_path = "data/train.csv"
holdout_path = "data/test.csv"
mlflow_uri = "http://127.0.0.1:8888"
mlflow_experiment_name = "Experiment #X"

# This df will be used for training and testing the model
train = pd.read_csv(train_path)

# This df is held aside for predicting and submitting to the competition
holdout = pd.read_csv(holdout_path)


# Set our tracking server uri for logging
def mlflow_setup():
    mlflow.set_tracking_uri(uri=mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)


mlflow_setup()

# Create the TitanicTransformer class, for cleaning up the passenger data


class TitanicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # drop unnecessary columns
        not_needed = ["PassengerId", "Name", "Cabin", "Age", "Ticket"]
        X = X.drop(not_needed, axis=1)

        # forward-fill missing embarcation data
        X["Embarked"] = X["Embarked"].ffill()

        # one-hot-encode embarcation data
        dummies = pd.get_dummies(X["Embarked"])
        X = X.join(dummies)
        X = X.drop(["Embarked"], axis=1)

        # encode sex as binary
        sex_encoder = LabelEncoder()
        X["Sex"] = sex_encoder.fit_transform(X["Sex"])

        # drop rows with null values
        X = X.dropna()

        # scale numeric cols
        numeric_cols = ["Pclass", "SibSp", "Parch", "Fare"]
        self.scaler.fit(X[numeric_cols])
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        return X


transformer = TitanicTransformer()

train_transformed = transformer.transform(train)
holdout_transformed = transformer.transform(holdout)

# Split the train dataset, for testing models

X = train_transformed.drop(["Survived"], axis=1)
y = train_transformed["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# Set parameters for MLFlow

params = {
    "solver": "lbfgs",
    "max_iter": 500,
    "multi_class": "auto",
    "random_state": 8888,
}

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy score is: " + str(accuracy_score(y_test, y_pred)))
print("Precision score is: " + str(precision_score(y_test, y_pred)))
print("Recall score is: " + str(recall_score(y_test, y_pred)))
print("f1 score is: " + str(f1_score(y_test, y_pred)))

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for Titanic data")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="titanic_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="titanic_v_001",
    )
print(model.coef_)

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

feature_names = X.columns

result = pd.DataFrame(X_test, columns=feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:20]
