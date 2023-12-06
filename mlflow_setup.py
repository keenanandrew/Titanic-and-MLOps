# Set up MLFlow with key information
import mlflow
from mlflow import MlflowClient

def ml_flow_setup():
    mlflow_uri = "http://127.0.0.1:8080"
    mlflow_experiment_name = "Titanic II: The Iceberg Returns"
    mlflow.set_tracking_uri(uri=mlflow_uri)
    mlflow.set_experiment(mlflow_experiment_name)