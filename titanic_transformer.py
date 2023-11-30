import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
