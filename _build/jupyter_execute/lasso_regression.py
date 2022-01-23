#!/usr/bin/env python
# coding: utf-8

# # Lasso Regression
# 
# # Datenimport

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  


# In[2]:


# seaborn settings
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

# Data Import
df = pd.read_csv("ready_data.csv")

# Change Datatypes from Data.ipynb
datatypes_toChange = {"priceCat": "category", "proximity": "category"}
df = df.astype(datatypes_toChange)

df


# # Vorgehensweise
# 
# In den nachfolgenden Punkten wird die Lasso Regression jeweils einmal mit **SK Learn** und **Statsmodels** durchgeführt.</br>
# Dabei sind unterschiedliche Vorgehensweisen bzgl. Einlesen der Daten zu beachten.
# 
# Vorarbeit wurde bereits in Data.ipynb erbracht.
# 
# Ferner wird in SK Learn der Hyperparameter Lambda mittels Cross-Validation ermittelt.

# # Modellierung in SK Learn Pipeline
# 
# Nachfolgend die Lasso-Regression in der SK-Learn Pipeline.
# 
# Zum Ende wird der Hyperparameter Lambda ausgegeben.

# In[3]:


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


# In[4]:


# create label
y = df['median_house_value']

# create features
X = df.drop(['sm_PpH','sm_RpH','median_house_value', 'share_bedrooms', "priceCat"], axis=1)

# create list of feature names
num_feature_names =  X.columns


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
X_train.head()


# In[6]:


col_log_ = ["rooms_per_household", "person_per_household"]
med_log_ = 'median_income'

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
    ])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
    ])

preprocessor = ColumnTransformer(transformers=[
    ('cat', categorical_transformer, selector(dtype_include="category")),
    ('logTrans', FunctionTransformer(np.log),col_log_),
    ('num', numeric_transformer, selector(dtype_exclude="category")),
    ])


# In[7]:


lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lasso', LassoCV(cv=5, random_state=10, max_iter=10000))
    ])


# In[8]:


# show pipeline
set_config(display="diagram")
# Fit model
lm_pipe.fit(X_train, y_train)


# In[9]:


reg = lm_pipe.named_steps['lasso']


# In[10]:


y_pred = lm_pipe.predict(X_test)


# In[11]:


print('R squared training set', round(lm_pipe.score(X_train, y_train)*100, 2))
print('R squared test set', round(lm_pipe.score(X_test, y_test)*100, 2))


# In[12]:


print("RMSE training set", mean_squared_error(y_train, lm_pipe.predict(X_train), squared=False))
print("RMSE test set", mean_squared_error(y_test, lm_pipe.predict(X_test), squared=False))


# In[13]:


print("Lambda:", reg.alpha_)


# Laut Ausgabe ist der Hyperparameter Lambda = 0.063103048006874
# 
# Der nachfolgende Plot sollte eigentlich eine Darstellung über die Folds bei Cross-Validation=5 erstellen.
# Eventuell fehlt die Anzeige wegen dem Abruf aus der Pipeline mit *reg = lm_pipe.named_steps['lasso']*. </br>
# Eine Kontrolle über den manuellen Weg mit dem ausgeworfenen Lambda ist möglich und wurde für das nachfolgende Statsmodels geplant. Die Umsetzung in SKLearn fehlte im Nachgang die Zeit.

# In[14]:


plt.semilogx(reg.alphas_, reg.mse_path_, ":")
plt.plot(
    reg.alphas_ ,
    reg.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)
plt.axvline(
    reg.alpha_, linestyle="--", color="k", label="alpha: CV estimate"
)

plt.legend()
plt.xlabel("alphas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")

ymin, ymax = 1000, 2500
plt.ylim(ymin, ymax);


# # Modellierung in Statsmodels
# 
# Nachfolgend wird die Vorbereitung für die Lasso Regression dargestellt. Dies erfolgt bis zur Standardisierung von allen nummerischen Spalten.</br>
# Mittlerweile gibt es eine Funktion bei Statsmodels, welche jedoch noch nicht ausgereift ist. Beispielsweise scheint die Funktion zur Ausgabe von üblichen Summary-Werten wie MSE und R2 zu fehlen.
# 
# Der Umfang wird in diesem Notebook aufgenommen, weil sehr viel Zeit in die Recherche geflossen ist. :-)

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.genmod.generalized_linear_model as slm
from statsmodels.tools.eval_measures import mse, rmse

sns.set_theme(style="ticks", color_codes=True)


# In[16]:


df_stats = pd.concat([df, dummies[["Ocean_Prox_INLAND","priceCat_below"]]], axis=1)
df_stats = df_stats.drop(columns=["Ocean_Prox", "priceCat"], axis=1)
numeric_feature_names = df_stats.drop(columns=["Ocean_Prox_INLAND", "priceCat_below", "median_house_value"],axis=1).columns


# In[5]:


train_dataset = df.sample(frac=0.8, random_state=0)
test_dataset = df.drop(train_dataset.index)


# In[6]:


train_dummies = pd.get_dummies(train_dataset[['priceCat', 'Ocean_Prox']], drop_first=True)
test_dummies = pd.get_dummies(test_dataset[['priceCat', 'Ocean_Prox']], drop_first=True)


# In[7]:


y_train_dataset = train_dataset["median_house_value"]
X_train_dataset = pd.concat([train_dataset.drop(columns=["median_house_value", 'priceCat','Ocean_Prox'], axis=1), train_dummies],axis=1)
y_test_dataset = test_dataset["median_house_value"]
X_test_dataset = pd.concat([test_dataset.drop(columns=["median_house_value", 'priceCat','Ocean_Prox'], axis=1), test_dummies],axis=1)

print(len(y_train_dataset))
print(len(X_train_dataset))


# In[8]:


from scipy import stats
#pd.concat([df, dummies[["Ocean_Prox_INLAND","priceCat_below"]]], axis=1)
X_train_dataset_z = pd.concat([X_train_dataset[numeric_feature_names].apply(stats.zscore), train_dummies[["Ocean_Prox_INLAND","priceCat_below"]]], axis=1)
X_test_dataset_z = pd.concat([X_test_dataset[numeric_feature_names].apply(stats.zscore), test_dummies[["Ocean_Prox_INLAND","priceCat_below"]]], axis=1)
X_train_dataset_z.info()


# # Modellierung
# Für statsmodel scheint noch keine ausgereifte Funktionalität bei Lasso-Regression zu bestehen.
# 
# Hier wurde das Alpha aus der SK Learn Pipeline exemplarisch eingesetzt.

# In[34]:


# Fit Model
#lasso_reg = slm.GLM(y_train_dataset,X_train_dataset_z).fit_regularized(method="",alpha=0.063103048006874, L1_wt=1)

