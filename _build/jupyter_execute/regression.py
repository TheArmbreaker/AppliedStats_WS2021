#!/usr/bin/env python
# coding: utf-8

# # Regression

# In this project, your goal is to build regression models of housing prices. The models should learn from data and be able to predict the median house price in a district (which is a population of 600 to 3000 people), given some predictor variables. 
# 

# # Data Import

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns  

# seaborn settings
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

df = pd.read_csv("ready_data.csv")
df


# In[2]:


datatypes_toChange = {"proximity": "category"}
df = df.astype(datatypes_toChange)
df = df.drop(["priceCat"], axis=1)


# # Vorgehensweise
# 
# In den nachfolgenden Punkten wird die Lineare Regression jeweils einmal mit **SK Learn** und **Statsmodels** durchgeführt.</br>
# Dabei sind unterschiedliche Vorgehensweisen bzgl. Einlesen der Daten zu beachten.
# 
# Vorarbeit wurde bereits in Data.ipynb erbracht.
# 
# Ferner wird zum Statsmodel-Anteil die Regression Diagnostic durchgeführt.

# # Modellierung in Statsmodels

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import mse, rmse
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.compat import lzip

sns.set_theme(style="ticks", color_codes=True)


# In[4]:


dummies = pd.get_dummies(df[["proximity"]], drop_first = True)


# In[5]:


df_stats = pd.concat([df, dummies[["proximity_INLAND"]]], axis=1)
df_stats = df_stats.drop(columns=["proximity", "person_per_household", "rooms_per_household"], axis=1)


# In[6]:


train_dataset = df_stats.sample(frac=0.8, random_state=0)
test_dataset = df_stats.drop(train_dataset.index)


# In[7]:


df_stats.info()


# In[8]:


# Fit Model
lm = smf.ols(formula='median_house_value ~ housing_median_age + median_income + sm_PpH + sm_RpH + share_bedrooms + proximity_INLAND' , data=train_dataset).fit()
# Full summary
lm.summary()


# ## Nicht-Normalverteilte Residuen
# 
# Obwohl Omnibus und Jarque-Bera die Null-Hypothese der Normalverteilten Residuen verwerfen, können wir wegen der Anzahl von 15T Beobachtungen (sehr großes n) von Normalverteilung ausgehen.
# 
# ## Correlation of Error Terms
# 
# Durbin Watson ist 2. Demnach liegt keine Autokorrelation der Fehlerterme vor.
# 
# ## Mulitkolliniarität
# 
# Eine hohe Condition Number ist ein Hinweis auf Mulitkolliniarität. Daher wird nachfolgen der Variance Inflation Factor betrachtet.
# 
# Im besten Fall sind die Werte = 1. Werte größer 5 sind sehr problematisch. </br>
# Im vorliegenden Modell sind hohe Werte für Rooms per Household und Share_Bedroom enthalten. Daher wird das Feature Share Bedroom entfernt.
# Das Summary zum neuen Modell fällt geringfühgig in R2, jedoch wir die Condition Number kleiner. Beim neuen VIF nimmt das Feature Rooms per Household deutlich ab.

# In[9]:


# choose features and add constant
features = add_constant(df[["housing_median_age", "median_income", "sm_PpH", "sm_RpH", "share_bedrooms"]])
# create empty DataFrame
vif = pd.DataFrame()
# calculate vif
vif["VIF Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
# add feature names
vif["Feature"] = features.columns

vif.round(2)


# In[10]:


# choose features and add constant
features = add_constant(df[["housing_median_age", "median_income", "sm_PpH", "sm_RpH"]])
# create empty DataFrame
vif = pd.DataFrame()
# calculate vif
vif["VIF Factor"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
# add feature names
vif["Feature"] = features.columns

vif.round(2)


# In[11]:


# Fit Model
lm = smf.ols(formula='median_house_value ~ housing_median_age + median_income + sm_PpH + sm_RpH + proximity_INLAND' , data=train_dataset).fit()
# Full summary
lm.summary()


# ## Influence und Outliers
# 
# Ausreißer und Punkte mit großen Einfluss auf die Regressionsgerade werden mittels Cook's Distance und Influence Plot identifiziert und aus dem Datensatz entfernt.
# 
# Dies verbessert den R2 von 62,5 auf 72,5.

# In[12]:


fig = sm.graphics.influence_plot(lm, criterion="cooks")
fig.tight_layout(pad=1.0)


# In[13]:


# obtain Cook's distance 
lm_cooksd = lm.get_influence().cooks_distance[0]

# get length of df to obtain n
n = len(train_dataset["median_income"])

# calculate critical d
critical_d = 4/n
print('Critical Cooks distance:', critical_d)

# identification of potential outliers with leverage
out_d = lm_cooksd > critical_d

# output potential outliers with leverage
print(train_dataset.index[out_d], "\n", lm_cooksd[out_d])
dont_want = np.array(train_dataset.index[out_d])


# In[14]:


a = len(train_dataset)

for i in dont_want:
    train_dataset = train_dataset.drop(index=i)

print("Gelöschte Datensätze nach Cook's distance:", a - len(train_dataset))


# In[15]:


lm = smf.ols(formula='median_house_value ~ housing_median_age + median_income + sm_PpH + sm_RpH + proximity_INLAND' , data=train_dataset).fit()
fig = sm.graphics.influence_plot(lm, criterion="cooks")
fig.tight_layout(pad=1.0)


# In[16]:


print("R2:",lm.rsquared)
print("R2_adj:",lm.rsquared_adj)


# ## Nicht-Linearität und Heteroskedastizität
# 
# Der Fitted vs Regression Plot zeigt jedoch einen möglichen nicht-linearen Zusammenhang in der Varianz der Residuen.</br>
# Dies wird mit dem **Breusch-Pagan Lagrange Multiplier test** überprüft. Homoskedastizität kann nicht angenommen werden, weil der p-value kleiner 0.05 ist und die Null-Hypothese nicht verworfen wird.
# 

# In[17]:


fig = sm.graphics.plot_partregress_grid(lm)
fig.tight_layout(pad=1.0)


# In[18]:


# fitted values
model_fitted_y = lm.fittedvalues

#  Plot
plot = sns.residplot(x=model_fitted_y, y='median_house_value', data=train_dataset, lowess=True, 
                     scatter_kws={'alpha': 0.5}, 
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

# Titel and labels
plot.set_title('Residuals vs Fitted')
plot.set_xlabel('Fitted values')
plot.set_ylabel('Residuals');


# In[19]:


name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
test = sm.stats.het_breuschpagan(lm.resid, lm.model.exog)
lzip(name, test)


# ## Validierung mit Testdaten

# In[20]:


# Add the regression predictions (as "pred") to our DataFrame
train_dataset['y_pred'] = lm.predict()
# Predict with Test_Dataset
test_dataset['y_pred'] = lm.predict(test_dataset)


# In[21]:


# MSE
print("MSE train dataset:",mse(train_dataset['median_house_value'], train_dataset['y_pred']))
print("RMSE train dataset:",rmse(train_dataset['median_house_value'], train_dataset['y_pred']))
print("MSE test dataset:",mse(test_dataset['median_house_value'], test_dataset['y_pred']))
print("RMSE test dataset:",rmse(test_dataset['median_house_value'], test_dataset['y_pred']))


# In[22]:


sns.residplot(x="y_pred", y="median_house_value", data=test_dataset, scatter_kws={"s": 80});


# # Modellierung in SK Learn Pipeline
# 
# Nachfolgend die Regression mit der Pipeline von SKLearn mit dem Austausch von Parametern zum Darstellen der Pipeline-Funktionalität.</br>
# Mit anpassen einer Liste von Predictor-Variablen können über die Pipeline zahlreiche Modelle ermittelt werden.
# 
# Zu beachten ist, dass die in Data.ipynb durchgeführten Log-Transformationen ebenfalls über die Pipeline erfolgen. D.h. getrennt für Trainings- und Testdaten.</br>
# Dies hilft bei der Standardizierung, welche hier ebenfalls für alle nummerischen Variablen vorgenommen wird.
# 
# Die Erkenntnisse aus Regression-Diagnostics könnten an dieser Stelle berücksichtigt werden. Dies wird ausgelassen, weil mir noch die Erfahrung in Python und die Zeit fehlt.</br>
# Da die R2-Werte um den gleichen Wert wie das Statsmodels-Modelle ohne Ausreißer-Entfernung (62,5), wird geschlussfolgert das eine Reduktion der Predictoren keine signifikante Verbesserung bringt.

# In[23]:


from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[24]:


features = ["median_income","proximity","rooms_per_household"]
col_log_ = ["rooms_per_household"]
med_log_ = 'median_income'
X = df[features]
y = df["median_house_value"]

X.info()


# In[25]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[26]:


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


# In[27]:


lm_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('lm', LinearRegression())
    ])


# In[28]:


# show pipeline
set_config(display="diagram")
# Fit model
lm_pipe.fit(X_train, y_train)
y_pred = lm_pipe.predict(X_train)
print("Feature:",[features])
print("R2:",r2_score(y_train, y_pred))
print("MSE:",mean_squared_error(y_train, y_pred))
print("RMSE:", mean_squared_error(y_train, y_pred, squared=False))


# In[30]:


# show pipeline
set_config(display="diagram")
# Fit model
lm_pipe.fit(X_train, y_train)
y_pred = lm_pipe.predict(X_test)
print("Test_Dataset")
print("Feature:",[features])
print("R2:",r2_score(y_test, y_pred))
print("MSE:",mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


# In[31]:


features = ["median_income","proximity","rooms_per_household", "person_per_household"]
col_log_ = ["rooms_per_household", "person_per_household"]
med_log_ = 'median_income'
X = df[features]
y = df["median_house_value"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm_pipe.fit(X_train, y_train)

y_pred = lm_pipe.predict(X_train)
print("Feature:",[features])
print("R2:",r2_score(y_train, y_pred))
print("MSE:",mean_squared_error(y_train, y_pred))
print("RMSE:", mean_squared_error(y_train, y_pred, squared=False))


# In[32]:


features = ["median_income","proximity","rooms_per_household", "person_per_household"]
col_log_ = ["rooms_per_household", "person_per_household"]
med_log_ = 'median_income'
X = df[features]
y = df["median_house_value"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm_pipe.fit(X_train, y_train)
print("Test_Dataset")
y_pred = lm_pipe.predict(X_test)
print("Feature:",[features])
print("R2:",r2_score(y_test, y_pred))
print("MSE:",mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))


# In[33]:


#	housing_median_age	median_income	median_house_value	priceCat	person_per_household	share_bedrooms	rooms_per_household#
features = ["median_income","proximity","rooms_per_household", "housing_median_age", "person_per_household"]
col_log_ = ["rooms_per_household","person_per_household"]
med_log_ = 'median_income'
X = df[features]
y = df["median_house_value"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm_pipe.fit(X_train, y_train)


y_pred = lm_pipe.predict(X_train)
print("Feature:",[features])
print("R2:",r2_score(y_train, y_pred))
print("MSE:",mean_squared_error(y_train, y_pred))
print("RMSE:", mean_squared_error(y_train, y_pred, squared=False))


# In[34]:


#	housing_median_age	median_income	median_house_value	priceCat	person_per_household	share_bedrooms	rooms_per_household#
features = ["median_income","proximity","rooms_per_household", "housing_median_age", "person_per_household"]
col_log_ = ["rooms_per_household","person_per_household"]
med_log_ = 'median_income'
X = df[features]
y = df["median_house_value"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm_pipe.fit(X_train, y_train)
print("Test_Dataset")
y_pred = lm_pipe.predict(X_test)
print("Feature:",[features])
print("R2:",r2_score(y_test, y_pred))
print("MSE:",mean_squared_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))

