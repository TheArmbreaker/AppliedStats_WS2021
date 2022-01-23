#!/usr/bin/env python
# coding: utf-8

# # Summary and Conclusion
# 
# ___

# Der Auftraggeber ist daraufhinzuweisen, dass die Modelle keine Zusammenhänge für Distrikte mit Median-Immobilienwerte über 500 T$ und für Immobilienbestand mit Median-Alter von über 52 Jahren lernen konnten.
# 
# <img src="https://kirenz.github.io/ds-python/_images/lifecycle.png" alt="alt text" width="400" align="center">
# 
# (Kirenz, J. 2021: https://kirenz.github.io/ds-python/docs/lifecycle.html)
# 
# ## Regression
# 
# Anhand des RMSE soll das beste Regressions-Modell ausgewählt werden.
# 
# **Ein wichtiger Hinweis.** Die nachfolgende Aussage ist für die Praxis nicht anwendbar, weil die Modelle unterschiedlich behandelt wurden und Erkenntnisgewinn bei der Neumodellierung teilweise nicht eingeganen ist. </br> Beispielsweise hat nur das OLS I eine Regression Diagnostics erfahren. Diesbezüglich ist interessant zu beobachten, dass die RMSE zu den Testdaten kaum voneinander abweichen. Es wäre interessant zu sehen wie viel besser die anderen Modelle nach dem Entfernen von Ausreißern in den Trainingsdaten funktionieren.
# 
# Der RMSE gibt an um wieviel Dollar die Ausgabewerte zu Immobilienpreise in den Distrikten im Mittel schwanken.</br>
# Anhand des RMSE sollte das OLS IV verwendet werden. Der Einfluss der Log-Transformationen auf nicht-lineare Modelle sollte bedacht und noch untersucht werden.
# 
# 
# |Modell	|RMSE_train  |RMSE_test	|
# |---	|---	|---	|
# |OLS I (sm)	|46.57	|60.70	|
# |OLS II (sk)	|64.68	|63.05	|
# |OLS III (sk)	|60.92	|59.19	|
# |OLS IV (sk)	|60.28	|58.24	|
# |Lasso (sk)	|59.32	|59.49	|
# |Spline (sk)	|57.98	|59.21	|
# |Spline (sm)	|73.49	|74.80	|
# 
# sm: Statsmodels</br>
# sk: SK Learn Pipeline

# ## Klassifikation
# 
# Ein Modell zur Klassifizierung von Immobilienwerten über 150 T$ wurde aufgestellt. Accurancy von 83 % ist ok.
# Über Kosten der falschen Zuordnung könnte sich noch Gedanken gemacht werden.
# 
# Das beste ermittelte Klassifikationsmodell hat eine Accuracy von 83 %. Wie im Notebook erläutert, wird Threshold und F1 Score bewusst nicht verfolgt.
# 
# Die **Precision** beträgt 83 % - d.h. 83% der als *below* (positiv) geschätzten Distrike sind positiv - haben einen Median-Immobilienwert von unter 150 T$ (inkl. Obergrenze).
# 
# 
# Der **Recall** beträgt 82 % - d.h. das Modell trifft 82% der tatsächlich positiven Distrike - also die tatsächlich ein Median-Immobilienwert von unter 150 T$ (inkl. Obergrenze) haben.
# 
