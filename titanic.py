# A script for the Titanic dataset
# For practicing mlflow, optuna and other MLOps libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # for visualisations during the data exploration stage
import matplotlib.pyplot as plt # another option for plots
import opendatasets as od # if needed for grabbing datasets

# preprocessing and analysis imports

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, \
                            confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss, f1_score 

# mlflow imports

import mlflow
from mlflow.models import infer_signature

# Import the data

train = pd.read_csv('./titanicdata/train.csv')
holdout = pd.read_csv('./titanicdata/test.csv')

# Set our tracking server uri for logging

mlflow.set_tracking_uri(uri="http://127.0.0.1:8888")

# Let's take a look.

train.head(10)


# # First things first.
# 
# ## What do I have so far? 
# 
# I have the **train** dataset, which is the unadapted information. It's a mix of datatypes, it's unnormalised, unregularised, contains both continous and categorical variables of different relevance. It contains features and outcome.
# 
# I have the **test** dataset, which is less than half the size (number of rows) of the train dataset. The main difference is that the test dataset is missing the final feature - in other words, the **outcome** - *Survival*. 
# 
# This has one (binary) variable - 1 for survival, 0 for death.
# 
# The objective?
# 
# Export a spreadsheet that predicts the Survival outcomes for all of the rows in the **test** dataset.
# 
# 

# # Explore the data

# In[2]:


# Let's have a closer look at the data.

print(train.dtypes) 
train.isna().sum()


# In[3]:


holdout.isna().sum()


# ## What should I do with each of these?
# 
# **PassengerID** will be the primary key. It doesn't need to change, and I should ensure it isn't factored into the model. Keep but ignore in model
# 
# **Survived** is the outcome - a binary variable, 1 for survived, 0 for icy dead people. SEPARATE INTO OWN DF.
# 
# **Pclass** is the passenger's class of travel. It's either 1, 2 or 3. This is an ordinal (ie ranked) categorical variable, which has its own encoding. Somehow.
# 
# **Name** is an object (why not a string?), which should really have no predictive power at all. DROP.
# 
# **Sex** is the sex of the passenger. It's an array of text, ['male','female'], which should be converted into the binary variable 1 and 0. ADAPT.
# **Age** is the age... as a float? With some fractions, and some NaNs just to add further complexity? CLEAN UP.
# 
# **SibSp** is an unusual one. It's an integer that contains the number of siblings and spouses that passenger had on board. How should this be handled? Should it be a ranked / ordinal categorical variable, or should it be compressed to a binary one? Does it matter at all? 
# 
# **Parch** is the number of parents and children on board. Really not sure why it's separate from SibSp, but not completely different.
# 
# **Ticket** is the serial(?) number of the ticket, plus some other 'information' such as the reference code for the port of purchase. It seems useless. DROP.
# 
# **Fare** is how much the individual's ticket cost, in (I assume) pound sterling. Pre-decimalisation, so who knows how that works. It probably doesn't matter as there can't be much more information than already provided in the travel class. CHECK FOR COLLINEARITY
# 
# **Cabin** is the cabin number; can it provide more information than travel class? Perhaps the location on the ship? STRIP OR DROP
# 
# **Embarked** is the embarkation port of that individual: **C**herbourg, **Q**ueenstown, **S**outhampton. CHECK FOR IMPACT.
# 
# ### Multivariate combinations to check
# 
# Could there be an interplay between Age, Parch and SibSp? For example, did people in big families survive / die disproport
# ionately? Or is that collinear with class of travel?
# 
# ### Nulls and gaps
# 
# So we have three columns with NaNs: Age, Cabin, Embarked.
# 
# Cabin has 687 NaNs of 891 rows, so it's useless and will need dropped as a column.
# 
# Age has 177 of 891 rows, so it'll need either imputed, or dropped. It's a bit too early for me to try the complicated kinds of imputation, and filling with the mean will be pretty useless.
# 
# Embarked has 2 missing, so I could just forward-fill, it probably won't even make a difference. Not clear yet if Embarked will have any impact anyway.
# 
# ## Conclusion:
# 
# * To be dropped: Name, Cabin, Age, Ticket
# * To be forward-filled: Embarked

# In[4]:


class TitanicTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns = None):
        self.scaler = StandardScaler()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # drop unnecessary columns
        not_needed = ['PassengerId', 'Name', 'Cabin', 'Age', 'Ticket']
        X = X.drop(not_needed, axis = 1)
        
        #forward-fill missing embarcation data
        X['Embarked'] = X['Embarked'].ffill()
        
        # one-hot-encode embarcation data
        dummies = pd.get_dummies(X['Embarked'])
        X = X.join(dummies)
        X = X.drop(['Embarked'], axis = 1)
        
        # encode sex as binary
        sex_encoder = LabelEncoder()
        X['Sex'] = sex_encoder.fit_transform(X['Sex'])
        
        # drop rows with null values
        X = X.dropna()
        
        # scale numeric cols
        numeric_cols = ['Pclass', 'SibSp', 'Parch', 'Fare']
        self.scaler.fit(X[numeric_cols])
        X[numeric_cols] = self.scaler.transform(X[numeric_cols])
        
        return X

transformer = TitanicTransformer()

train_transformed = transformer.transform(train)
holdout_transformed = transformer.transform(holdout)


# In[5]:


# Now to test WITHIN the train dataset
# Separate into features and outcome
# Split the train dataset

X = train_transformed.drop(['Survived'], axis = 1)
y = train_transformed['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state=42)


# In[6]:


plt.subplots(figsize=(8, 5))
sns.heatmap(X.corr(), annot=True, cmap="RdYlGn")
plt.show()


# In[7]:


# Set parameters for MLFlow

mlflow.set_experiment("MLflow Quickstart")


params = {
    "solver": "lbfgs",
    "max_iter": 500,
    "multi_class": "auto",
    "random_state": 8888,
}


# In[8]:


model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy score is: '+ str(accuracy_score(y_test, y_pred)))
print('Precision score is: '+ str(precision_score(y_test, y_pred)))
print('Recall score is: '+ str(recall_score(y_test, y_pred)))
print('f1 score is: '+ str(f1_score(y_test, y_pred)))



# In[9]:


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


# In[10]:


print(model.coef_)


# In[11]:


loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

feature_names = X.columns

result = pd.DataFrame(X_test, columns=feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:20]


# In[12]:


# output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
# output.to_csv('submission.csv', index=False)
# print("Your submission was successfully saved!")


