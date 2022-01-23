#!/usr/bin/env python
# coding: utf-8

# # Introduction
# ---

# # Vorgehensweise in dieser Arbeit
# 
# Die Arbeit orientiert sich am Data Science Lifecycle, welcher nachfolgend dargestellt wird.
# 
# Die blau markierte Phase der Planung wird zur Vollständigkeit auf dieser Seite in einem rudimentären Niveau durchgegangen.</br>
# 
# Das Feature Engineering wird gleichzeitig auf Test- und Trainingsdaten angewendet, wenn sichergestellt ist das keine Informationen des jeweils anderen Datasets in die Modelle einfließen. Beispielsweise kann eine Wurzel-Transformation sofort erfolgen, weil kein Mittelwert oder sonstiger Lagewert verwendet wird. Im Gegensatz dazu wird eine Standardisierung, welche den Mittelwert verwendet, bei Bedarf separat auf das jeweilige Datensets angewendet.
# 
# <img src="https://kirenz.github.io/ds-python/_images/lifecycle.png" alt="alt text" width="600" align="center">
# 
# (Kirenz, J. 2021: https://kirenz.github.io/ds-python/docs/lifecycle.html)

# ## Planing
# 
# Zur Bewertung von Immobilienpreisen in bestimmten Distrikten sollen mehrere Modelle erstellt werden.
# 
# Geschätzt werden soll der Median-Preis von Immobilien mit folgenden Modellen.
# 
# * OLS-Regression
# * Lasso-Regression
# * Regression mit Splines
# 
# Ferner wird klassifiziert, ob die Median-Preise in Distrikten über der Preisschwelle von 150T$ liegen.
# 
# * Logistische Regression

# ## Variablen
# 
# Der zur Verfügung gestellte Datensatz enthält folgende Werte.
# 
# ---
# __longitude und latidue__: Koordinaten zur Orientierung</br>
# __housing_median_age:__ Median age of a house within a district; a lower number is a newer building</br>
# __total_rooms:__ Total number of rooms within a district </br>
# __total_bedrooms:__ Total number of bedrooms within a district</br>
# __population:__ Total number of people residing within a district</br>
# __households:__ Total number of households, a group of people residing within a home unit, for a district</br>
# __median_income:__ Median income for households within a district of houses (measured in tens of thousands of US Dollars)</br>
# __median_house_value:__ Median house value within a district (measured in US Dollars)</br>
# __ocean_proximity:__ Location of the district</br>
# __price_category:__ Indicator variable made from median_house_value (if median house value is below or above 150000)</br>
# 
# ---
# 
# Mit Hilfe der nachfolgend dargestellten Variablen werden die Modelle trainiert und getestet.
# In den Modellen werden die Variabelen *price_categorie* und *median_house_value* die Responsewerte darstellen.</br>
# Ferner liegt mit *ocean_proximity* eine weitere Variable zur Strand- oder Ozean-Nähe vor.</br>
# Die restlichen Variablen sind absolute oder Medianwerte.

# In[1]:


import pandas as pd
df = pd.read_csv("project_data.csv")


# In[2]:


df.groupby(by=["price_category","ocean_proximity"]).describe().T

