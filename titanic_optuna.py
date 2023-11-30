# A script for the Titanic dataset
# For practicing mlflow, optuna and other MLOps libraries
from functools import \
    partial  # to solve scoping problem when supplying more params' to objective function

import mlflow
# Optuna
import optuna
import pandas as pd
from mlflow.models import infer_signature
from optuna.samplers import TPESampler
from sklearn import datasets
# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import cross_val_score, train_test_split

from titanic_transformer import TitanicTransformer

# Import the data

train_path = "data/train.csv"
holdout_path = "data/test.csv"
mlflow_uri = "http://127.0.0.1:8880"
mlflow_experiment_name = "Titanic sandbox"

# This df will be used for training and testing the model
train = pd.read_csv(train_path)

# This df is held aside for predicting and submitting to the competition
holdout = pd.read_csv(holdout_path)


# Set our tracking server uri for logging
def mlflow_setup():
    mlflow.set_tracking_uri(uri=mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)


mlflow_setup()

transformer = TitanicTransformer()

train_transformed = transformer.transform(train)
holdout_transformed = transformer.transform(holdout)

# ====== OPTUNA ADDED HERE =======


def objective(trial, n_folds, X, y):
    """Objective function for tuning logistic regression hyperparameters"""

    # in Optuna, the search space is inside the function definition
    # Why is that better?

    params = {
        "warm_start": trial.suggest_categorical("warm_start", [True, False]),
        "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False]),
        "tol": trial.suggest_float("tol", 0.00001, 0.0001),
        "C": trial.suggest_float("C", 0.05, 2.5),
        "solver": trial.suggest_categorical(
            "solver", ["newton-cg", "lbfgs", "liblinear"]
        ),
        "max_iter": trial.suggest_categorical("max_iter", range(10, 500)),
    }
    # Perform n_fold cross validation with hyperparameters
    clf = LogisticRegression(**params, random_state=42)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring="f1_macro")

    # Extract the best score
    max_score = max(scores)

    # Loss must be minimized
    loss = 1 - max_score

    # Dictionary with information for evaluation
    return loss


# ======== OPTUNA ENDS HERE=======





# Split the train dataset, for testing models

n_folds = 5
X = train_transformed.drop(["Survived"], axis=1)
y = train_transformed["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)


from optuna.samplers import TPESampler

study = optuna.create_study(direction="minimize", sampler=TPESampler())
study.optimize(
    partial(objective, n_folds=n_folds, X=X_train, y=y_train), n_trials=16
)

print(study.best_trial.params)

# model = LogisticRegression()

# model.fit(X_train, y_train)

# y_pred = clf.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)


# ===== DROP MLFLOW FOR NOW =====

# with mlflow.start_run():
#     # Log the hyperparameters
#     mlflow.log_params(params)

#     # Log the loss metric
#     mlflow.log_metric("accuracy", accuracy)

#     # Set a tag that we can use to remind ourselves what this run was for
#     mlflow.set_tag("Training Info", "Basic LR model for Titanic data")

#     # Infer the model signature
#     signature = infer_signature(X_train, model.predict(X_train))

#     # Log the model
#     model_info = mlflow.sklearn.log_model(
#         sk_model=model,
#         artifact_path="titanic_model",
#         signature=signature,
#         input_example=X_train,
#         registered_model_name="titanic_v_001",
#     )
# print(model.coef_)

# loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

# predictions = loaded_model.predict(X_test)

# feature_names = X.columns

# result = pd.DataFrame(X_test, columns=feature_names)
# result["actual_class"] = y_test
# result["predicted_class"] = predictions

# result[:20]
