#!/usr/bin/env python
# coding: utf-8

# # Natural Spline
# 
# # Data Import

# In[1]:


import pandas as pd

df = pd.read_csv('ready_data.csv')
df


# In[2]:


df = df.drop(["priceCat"], axis=1)


# # Vorgehensweise
# 
# In den nachfolgenden Punkten wird die Regression eines Natural Splines jeweils einmal mit **SK Learn** und **Statsmodels** durchgeführt.</br>
# 
# Vorarbeit wurde bereits in Data.ipynb erbracht.
# 
# SK Learn Pipeline verwendet eine Lasso-Regression mit dem ermittelten Hyperparameter aus *lasso_regression.ipynb*. Eventuell handelt es sich dadurch nicht um einen **Natural** Spline.

# # Modellierung in SK Learn Pipeline
# 
# Für den Spline wird der Hyperparameter aus der Aufgabe zur Lasso-Regression verwendet. Es handelt sich nicht um einen Natural Spline</br>
# Lambda: 0.063103048006874

# In[3]:


from sklearn.metrics import mean_squared_error

# create function to obtain model mse
def model_results(model_name):

    # Training data
    pred_train = reg.predict(X_train)
    mse_train = round(mean_squared_error(y_train, pred_train, squared=True),4)
    rmse_train = round(mean_squared_error(y_train, pred_train, squared=False),4)

    # Test data
    pred_test = reg.predict(X_test)
    mse_test =round(mean_squared_error(y_test, pred_test, squared=True),4)
    rmse_test =round(mean_squared_error(y_test, pred_test, squared=False),4)

    # Print model results
    result = pd.DataFrame(
        {"model": model_name, 
        "mse_train": [mse_train],
        "rmse_train": [rmse_train],
        "mse_test": [mse_test], 
        "rmse_test": [rmse_test],
        }
        )
    
    return result;


# In[4]:


X = df[["housing_median_age", "median_income", "sm_RpH", "sm_PpH", "proximity",]] #sm_ verwendet, um nicht nochmal den FunctionTransformer zuschreiben.
y = df["median_house_value"]


# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# In[6]:


from sklearn.preprocessing import SplineTransformer, OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer, ColumnTransformer

lasso = Lasso(alpha=0.063103048006874) # Bestes Alpha aus Aufgabe zur Lasso Regression
column_trans = ColumnTransformer(remainder='passthrough', transformers=[('onehotencoder', OneHotEncoder(),['proximity']), ('standscal', StandardScaler(),['median_income'])])

reg = make_pipeline(column_trans, SplineTransformer(n_knots=4, degree=3), lasso)
reg.fit(X_train, y_train)

y_pred = reg.predict(X_train)


# In[7]:


model_results(model_name = "spline")


# # Modellierung in Statsmodels & Patsy

# In[8]:


from patsy import dmatrix
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[9]:


X = df[["median_income"]] #sm_ verwendet, um nicht nochmal den FunctionTransformer zuschreiben.
y = df["median_house_value"]


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# In[11]:


transformed_x3 = dmatrix("cr(train, df=3)", {"train": X_train},return_type='dataframe')


# In[12]:


reg = sm.GLM(y_train, transformed_x3).fit()


# In[13]:


# Training data
pred_train = reg.predict(dmatrix("cr(train, df=3)", {"train": X_train}, return_type='dataframe'))
mse_train = mean_squared_error(y_train, pred_train, squared=True)
rmse_train = mean_squared_error(y_train, pred_train, squared=False)

# Test data
pred_test = reg.predict(dmatrix("cr(test, df=3)", {"test": X_test}, return_type='dataframe'))
mse_test = mean_squared_error(y_test, pred_test, squared=True)
rmse_test = mean_squared_error(y_test, pred_test, squared=False)

# Save model results
model_results_ns = pd.DataFrame(
    {
    "model": "Natural spline (ns)", 
    "mse_train": [mse_train],  
    "rmse_train": [rmse_train],
    "mse_test": [mse_test], 
    "rmse_test": [rmse_test],
    })

model_results_ns

