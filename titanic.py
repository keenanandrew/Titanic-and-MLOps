# A script for the Titanic dataset
# For practicing mlflow, optuna and other MLOps libraries

from pprint import pprint

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from mlflow_setup import ml_flow_setup

from titanic_transformer import TitanicTransformer

# Access the data and make it available as dataframes

train_path = "data/train.csv"
holdout_path = "data/test.csv"

train = pd.read_csv(train_path)
holdout = pd.read_csv(holdout_path)

# get MLFlow going
ml_flow_setup()

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
    "max_iter": 400,
    "multi_class": "auto",
    "random_state": 8888,
}

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("Recall", recall)
    mlflow.log_metric("F1 Score", f1)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for Titanic data")

    # Infer the model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="mlruns",
        signature=signature,
        input_example=X_train,
        registered_model_name="titanic_v_002",
    )
# print(model.coef_)

# loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# predictions = loaded_model.predict(X_test)

# feature_names = X.columns

# result = pd.DataFrame(X_test, columns=feature_names)
# result["actual_class"] = y_test
# result["predicted_class"] = predictions

# result[:20]
