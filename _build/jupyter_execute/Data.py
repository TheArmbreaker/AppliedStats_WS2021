#!/usr/bin/env python
# coding: utf-8

# # Data / Stats / EDA

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats


# # Data Inspection and Transformation
# 
# Das Datenset enthält 10 Spalten mit unterschiedlichen Datentypen.
# 
# - Für *housing_median_age* und *median_house_value* wird eigentlich ein Int oder Float-Datentyp erwartet, weshalb die Daten nachfolgend ausgelesen und ausgewertet werden.
# - Aus der Beschreibung des Datensatzes ist bekannt, dass es sich bei *ocean_proximity* und *price_category* um Categoriale Variablen handelt. 
# - Longitude und Latitude werden entfernt, weil sie für die Aufgabenstellung unerheblich erscheinen.

# In[2]:



df = pd.read_csv("project_data.csv")
df.info()


# In[3]:


df = df.drop(["longitude","latitude"], axis=1)


# ### Bereinigung Datensatz
# 
# Bei *housing_median_age* und *median_house_value* wurde ein einzelner Datensatz mit untypischen Eingaben identifiziert. Die Bereinigung erfolgt manuell, weil es nur ein Datensatz ist. Anschließenden werden die korrekten Datentypen vergeben für die neuen Spalten dem Datentyp *float64* zugeordnet. Bei dieser Gelegenheit erhalten die Categorial Variablen auch den Datentyp *category*.

# In[4]:


amt_nofloat_age = 0
List_age = []
amt_nofloat_value = 0
List_value = []
for i in range(len(df["housing_median_age"]) - 1):
    try:
        float(df["housing_median_age"].values[i])
    except ValueError:
        amt_nofloat_age += 1
        List_age.append(i)

for j in range(len(df["median_house_value"]) - 1):
    try:
        float(df["median_house_value"].values[j])
    except ValueError:
        amt_nofloat_value += 1
        List_value.append(j)

print("Nicht-Float-Werte in Spalte housing_median_age:", amt_nofloat_age, "von", len(df["housing_median_age"]),". In folgenden Zeilen:",List_age)
print("Nicht-Float-Werte in Spalte median_house_value:", amt_nofloat_value, "von", len(df["median_house_value"]),". In folgenden Zeilen:", List_value)

df.iloc[0]


# In[5]:


df.loc[0,"housing_median_age"] = 41.0
df.loc[0,"median_house_value"] = 452600.0
df.iloc[0]


# In[6]:


datatypes_toChange = {"housing_median_age": "float64", "median_house_value": "float64", "ocean_proximity": "category", "price_category": "category"}
df = df.astype(datatypes_toChange)


# ### Fehlende Werte
# 

# Nachfolgend wird dargestellt, dass es in der Spalte *total_bedrooms* 207 Einträge fehlen.
# Die Datensätze könnten mit dem Median oder einer sonstigen Strategie aufgefüllt werden. Dies entfällt hier.
# Hinweis: Datensätze dürfen nur aufgefüllt werden, wenn kein weiterer Split mehr erfolgt. Dadurch soll verhindert werden, dass Informationen aus anderen Trainings/Test-Sets enthalten sind.

# In[7]:


print(df.isnull().sum())


# In[8]:



df_a = df.isnull()
df = df.dropna() #Dropen der Zeilen mit Nullwert
df_b = df.isnull()


# In[9]:


fig, axes = plt.subplots(1,2, figsize=(18,6))
sns.heatmap(df_a,yticklabels=False,cbar=False,cmap='viridis', ax=axes[0])
sns.heatmap(df_b,yticklabels=False,cbar=False,cmap='viridis', ax=axes[1])
plt.show()
plt.tight_layout()


# Nachdem die technische Inspektion abgeschlossen ist, werden nochmals Informationen zum Dataframe abgerufen.
# * Uns stehen 20433 Datensätze zur Verfügung.
# * Die Datentypen entsprechen den erwarteten Datentypen

# In[10]:


df.info()


# ___
# 
# # Statistik, Exploratory Data Analysis und Feature Engineering
# 
# ## Median_House_Value & Preiskategorie
# 
# Zunächst betrachten wir unsere Zielvariablen (Response-Values/Variablen).
# 
# Dazu werden die *median_house_values* nach Preiskategorie gruppiert und erste Lagemaße ausgegeben.
# 
# In der Preiskategorie liegt bereits das 2. Quantil über 150T$. Dies ist der Schwellenwert zwischen *below* und *above* aus der Aufgabenstellung. D.h. die Werte in der Spalte *price_categore* müssen neu ermittelt werden. Die sich überschneidende Verteilung der Preiskategorien verdeutlicht auch der nachstehende Boxplot, welcher eine Visualisierung der Lagemaße darstellt.
# 
# Die neue Spalte für *price_category* wird *priceCat* genannt. Die alte Spalte wird entfernt. Siehe Boxplot.

# In[11]:


df.groupby(by="price_category")["median_house_value"].describe().T


# In[12]:


df["priceCat"] = np.where(df.median_house_value >= 150000, "above", "below")
df["priceCat"] = df["priceCat"].astype("category")


# In[13]:


sns.boxplot(x="price_category",y="median_house_value",data=df)
plt.title("Verteilung des median_house_value über die Preiskategorien \n vor Bereinigung")


# In[14]:


sns.boxplot(x="priceCat",y="median_house_value",data=df)
plt.title("Verteilung des median_house_value über die Preiskategorien \n nach Bereinigung")


# Die neue Verteilung zeigt deutlich, dass der maximale Wert - in dem Fall der obere Whisker - von **below** dort liegt wo der minimale Wert von **above** anfängt.
# 
# Nachfolgend wird die neue tabellarische Übersicht dargestellt.
# 
# Der maximale Wert im Datensatz beträgt 500 T$ und im Mittel schwanken die Immobilienwerte der *above*-Kategorie mehr als doppelt so stark wie die in der Kategorie *below*.
# Außerdem sind die höherpreisigen Immobilien im Datensatz deutlich stärker vertreten (Anzahl).

# In[15]:


df = df.drop(["price_category"], axis=1) #Löschen der alten Preiskategorie
df.groupby(by="priceCat")["median_house_value"].describe().T


# Nachfolgend wird die Verteilung von *median_house_value* nach Preiskategorie geplotet.
# 
# Für die Kategorie *below* (gelb) zeigt sich eine Verteilung, welche dank zentralen Grenzwertsatz als normalverteilt angesehen werden kann - obwohl sie nicht perfekt verläuft.
# Problematischer ist die Verteilung von *above* (blau).
# 
# Verteilung *above*:
# 
# * Die Verteilung ist linkssteil (rechtsschief)
# * Bei ca. 300 T$ bildet sich ein leichter Anstieg.
# * Bei 500 T$ entsteht eine deutliche Wölbung.
# 
# Der Scatterplot auf der rechten Seite zeigt eine vertikale Linie für den Bereich um 500 T$ Immobilienwert, verteilt über sämtliche Einkommen. Dies ist eine kritische Datenlage, weil das Ende einer Regressionsgeraden in alle Richtungen ausschwanken kann. Eventuell besteht kein linearer Zusammenhang oder eine weitere Verteilung ist in der Preiskategorie *above* enthalten.
# 
# Dort wo sich im linken Plot das Plateau bildet, sind beim Scatterplot weitere vertikal verlaufende Linien zu beobachten.

# In[16]:


fig, axes = plt.subplots(1,2, figsize=(18,6))
sns.kdeplot(hue="priceCat",x="median_house_value", data=df, ax=axes[0])
sns.scatterplot(hue="priceCat",x="median_house_value",y="median_income", data=df, ax=axes[1])
plt.show()
plt.tight_layout()


# In[17]:


df_exp = df[df.median_house_value > 480000]
df_exp.describe()


# In[18]:



fig, axes = plt.subplots(1,3, figsize=(25,6))
sns.boxplot(hue="priceCat",x="median_house_value",data=df_exp, ax=axes[0])
sns.scatterplot(hue="ocean_proximity", x="median_house_value",y="median_income", data=df_exp, ax=axes[1])
sns.kdeplot(hue="ocean_proximity", x="median_house_value", data=df_exp, ax=axes[2])
plt.show()
plt.tight_layout()


# ### Feature Transformation für *median_house_value*
# 
# Es wurde das Phänomen des "Sammelns" von vielen Werten am oberen Rand erkannt. Wie oben beschrieben, scheinen die anderen Variablen gleichmäßiger in dem extrahierten Bereich vorzukommen.
# 
# Unsere Aufgabenstellung konzentriert sich jedoch maßgeblich auf den Immobilienwert, weshalb das Phänomen entfernt wird. D.h. aus den Rohdaten werden nur solche Daten behalten, welche nicht zu diesem Phänomen beitragen.</br>
# Dies kann unterstützt werden, weil es sich um Daten am oberen Rand handelt. Die oben angesprochenen "kleineren" Phänomene bleiben zunächst erhalten.
# 
# Zur Bereinigung wird das **1. Quartil des extrahierten Datensatzes  - 1$ (500T$)** als Obergrenze auf die Rohdaten angewendet. Dadurch die kleineren Datenpunkte für die Modellierung erhalten bleiben. Jedoch verstärkt sich die Wölbung der kleineren Phänomene entsprechend. Der eine Doller wurde zusätzlich abgezogen, weil bei der Verwendung des Quartils viele Werte "nach unten" stehen geblieben sind und die vertikale Linie nicht verschwand. Der Code ist im nächsten Block ausgegraut zum Replizieren.
# 
# Mit der nachfolgenden Ausgabe wird gezeigt, dass der höchste Immobilienwert im Datensatz 500 T$ beträgt und der niedrigste Wert knappe 15 T$. Im Mittel weichen die Immobilienwerte 98 T$ voneinander ab, wobei hier eine gemeinsame Verteilung für die Preiskategorie ausgegeben wird. Wesentliche Erkenntnis ist, dass das spätere Modelle für die Schätzung von Immobilien mit einem Wert von 500 T$ ihre Güte verlieren und bestenfalls nur in der Range mit Ausgabewerten zwischen 15 T$ und 500 T$ verwendet werden sollten. Beim Deployment müssen die Anwender daraufhingewiesen werden.

# In[19]:


#df = df[df.median_house_value < np.percentile(df_exp.median_house_value, q=[25])[0]] #25th Percentile oder auch 1. Quartil des Anomalie-Boxplot werden als Obergrenze verwendet.
df = df[df.median_house_value < 500000]
df["median_house_value"].describe()


# In[20]:



fig, axes = plt.subplots(1,2, figsize=(18,6))
sns.kdeplot(hue="priceCat",x="median_house_value", data=df, ax=axes[0])
sns.scatterplot(hue="priceCat",x="median_house_value",y="median_income", data=df, ax=axes[1])
plt.show()
plt.tight_layout()


# ## Median Income & Preiskategorie
# 
# ### EDA von *median_income* und Transformation von *median_house_value*
# 
# Wie im oberen Scatterplot bereits erkannt werden kann, gibt es einen deutlichen Unterschied zwischen der Verteilung von *median_income* zwischen den Preiskategorien.</br>
# 
# Die nachfolgende Tabelle zeigt ein Median-Einkommen zwischen 0.5 T$ und 15 T$ pro Monat. Wie durch nachfolgenden Boxplot und Dichteplot deutlich wird, weichen die Verteilungen der Preiskategorien voneinander ab.</br> Im Mittel schwanken die Einkommen für *below* um 1 T$ und für *above* um 1.5 T$.
# 
# Die unterschiedliche Skalierung zwischen Einkommen (in Tausend Dollar) und Immobilienwert (Absolute Dollar) kann für Probleme bei Modellschätzungen sorgen. Hier wird eine Transformation angeraten.
# 
# Im Rahmen der Projektarbeit wurde über sqrt- oder log-Transformationen nachgedacht (vgl. Codezeilen), um hohe Werte zu stauchen. Stattdessen wird die Zeile *median_house_value* mit dem Faktor 1000 dividiert. Dies hat keinen Einfluss auf die Verteilung.

# In[21]:


fig, axes = plt.subplots(1,2, figsize=(18,6))
sns.kdeplot(hue="priceCat",x="median_income", data=df, ax=axes[0])
sns.boxplot(y="priceCat", x="median_income",data=df, ax=axes[1])
plt.show()
plt.tight_layout()


# In[22]:


print("\n Median_Income:\n",df.groupby(by="priceCat")["median_income"].describe().T, "\n Median_House_Value:\n", df["median_house_value"].describe())
#print("Modus:\n",df.mode().groupby(by="priceCat"))
#print("Modus:\n",df[["priceCat","median_income"]].groupby('priceCat').apply(pd.DataFrame.mode).reset_index(drop=True).T)
#df[["priceCat","median_income"]].groupby('priceCat').apply(pd.DataFrame.mode).reset_index(drop=True)


# In[23]:


#df['median_income'] = np.log(df['median_income']) #Log-Transformation
#df['median_income'] = np.log(df['median_income']) #Sqrt-Transformation
df.loc[:,'median_house_value'] = df.loc[:,'median_house_value'] / 1000 #Transformation by Value


# In[24]:


df["median_house_value"].describe()


# In[25]:


#fig, axes = plt.subplots(1,2, figsize=(18,6))
#sns.kdeplot(hue="priceCat",x="median_income", data=df, ax=axes[0])
#sns.boxplot(y="priceCat", x="median_income",data=df, ax=axes[1])
#plt.show()
#plt.tight_layout()


# ## Housing Age & Preiskategorie
# 
# * Die Verteilung der Median-Gebäudealter ist nicht besonders schön verteilt, aber kann als normal verteilt angenommen werden.
# * Die Preiskategorie überschneiden sich.
# * Die Lage-Kategorien überschneiden sich weitestgehend. Lediglich die Bay Area und Inseln fallen mit **im Vergleich** alten Gebäudebestand auf.
# 
# ___
# 
# * Die Median-Gebäudealter liegen zwischen einem und 52 Jahren.
# * Die nicht erklärte Standardabweichung beträgt für beide Preiskategorien 12-13 Jahre bei einem Mittelwert von jeweils 28 Jahren.
#     * D.h. im Mittel schwankt das Gebäudealter zwischen 15 und 41 Jahren.
# 
# Es gilt die gleiche Bedingung wie beim Immobilienwert. Unser Modell wird den Zusammenhang zwischen Wert und einem Median-Alter > 52 Jahre nicht lernen können.
# Viel wichtiger ist die inverse Beziehung, welche die Modellschätzung beeinträchtigen kann. Einigen Modellen fällt es rechnerisch leichter hohe Prediktoren mit hohe Response Variablen zu verbinden.
# Eine Transformation könnte angebracht sein.
# 
# Die Klassifikation ist nicht davon betroffen, weil sich die Verteilungen nach Preiskategorie überschneiden.</br>
# Insbesondere die linearen Regressionsmodelle würden sich über eine Anpassung "freuen". Nach Blick auf den Scatterplot zwischen Einkommen und Alter wird dieses Problem verworfen und es erfolgt keine Transformation. Den geringen Zusammenhang bestätigt auch die nachfolgende Korrelationsanalyse.
# 

# In[26]:


df.groupby(by="priceCat")["housing_median_age"].describe().T


# In[27]:


fig, axes = plt.subplots(1,3, figsize=(21,6))
sns.kdeplot(hue="priceCat",x="housing_median_age", data=df, ax=axes[0])
sns.boxplot(y="priceCat", x="housing_median_age",data=df, ax=axes[1])
sns.boxplot(y="ocean_proximity", x="housing_median_age",data=df, ax=axes[2])
plt.show()
plt.tight_layout()


# In[28]:


sns.scatterplot(x="housing_median_age",y="median_house_value",data=df, hue="priceCat")
#sns.histplot(x="housing_median_age", y="median_house_value", data=df)


# ## Korrelationsanalyse mit Feature Engineering
# 
# ### Korrelationsanalyse und neue Features
# 
# Die Pearson-Korrelationskoeffizienten zeigen starke Abhängigkeiten zwischen den Variablen *total rooms*, *total bedrooms*, *population* und *households*.
# 
# Zur Vermeidung von Multi-Kolinearität dürfen die Werte nicht verwendet werden. Jedoch können damit weitere Features ermittelt werden.
# - *Population* und *Haushalt* lassen sich zur Ermittlung der durchschnittlichen *Personen pro Haushalt* verwenden.
# - Ein mittlerer Wert zu *Räume pro Haushalt* und zum *Schlafzimmer-Anteil* ermöglichen einen Schluss auf die Hausgröße, welche wiederum Einfluss auf den Wert des Hauses haben könnte.
# 
# Bei Letzteren muss dennoch auf Kolinearität geachtet werden. Da für die Anzahl an Schlafzimmern und Anzahl an Räumen ein Zusammenhang unterstellt werden kann.
# 
# Ferner wird auch eine deutliche Korrelation zwischen *median_income* und *median_house_value* erkennbar.

# In[29]:


# Calculate correlation using the default method ( "pearson")
corr = df.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
fig, axes = plt.subplots(1,1, figsize=(40,10))
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, square=True, annot_kws={"size": 12});


# In[30]:


df["person_per_household"] = df["population"]/df["households"]
df["share_bedrooms"] = df["total_bedrooms"]/df["total_rooms"]
df["rooms_per_household"] = df["total_rooms"]/df["households"]
df = df.drop(["total_rooms","total_bedrooms","population","households"],axis=1)
df


# In[31]:


# Calculate correlation using the default method ( "pearson")
corr = df.corr()
# optimize aesthetics: generate mask for removing duplicate / unnecessary info
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
# Generate a custom diverging colormap as indicator for correlations:
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Plot
fig, axes = plt.subplots(1,1, figsize=(40,8))
sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, square=True, annot_kws={"size": 12});


# ## Überblick zu neuen Features und Transformation
# 
# Nachfolgend werden die Verteilungen auf für die neuen Features betrachtet.
# 
# * *persons_per_household* hat einige ungewöhnliche Ausreißer mit im Mittel über Tausend Personen pro Haushalt in einem Distrikt.
# * Das Feature *person_per_household* wird mit dem 4-fachen der Interquartilsabweichung von Ausreißern nach Oben bereinigt.
# * Die Features *rooms_per_household* und *person_per_household* werden mit Log-Transformation bearbeitet, um weiteren starke Ausreißern weniger Gewicht zu geben.
#     * Das Datenset wird um zwei Spalten erweitert: *sm_PpH* und *sm_RpH*
#     * Die neuen Spalten enthalten die Log-Transformation für Modellierung mit Statsmodels
#     * Die ursprünglichen Spalten bleiben erhalten für die Verarbeitung in der SKLearn Pipeline und werden bei Bedarf für die Splines verwendet.
# 
# Das Feature *share_bedrooms* ist wegen seiner Beschaffenheit (Anteil) bereits zwischen 0 und 1 normiert und muss nicht weiter angepasst werden.
# </br>Die Verteilungen nach Preiskategorie sind nicht sehr unterschiedlich.

# In[32]:


hue_ = "priceCat"
fig, axes = plt.subplots(3,1, figsize=(10,18))
sns.kdeplot(hue=hue_ ,x="person_per_household", data=df, ax=axes[0])
sns.kdeplot(hue=hue_ , x="share_bedrooms",data=df, ax=axes[1])
sns.kdeplot(hue=hue_ , x="rooms_per_household",data=df, ax=axes[2])
plt.show()
plt.tight_layout()


# In[33]:


hue_ = "priceCat"
fig, axes = plt.subplots(3,1, figsize=(10,18))
sns.scatterplot(y="median_house_value", x="person_per_household", data=df, ax=axes[0])
sns.scatterplot(y="median_house_value", x="share_bedrooms",data=df, ax=axes[1])
sns.scatterplot(y="median_house_value", x="rooms_per_household",data=df, ax=axes[2])
plt.show()
plt.tight_layout()


# In[34]:


sns.boxplot(data=df, x="person_per_household")
df.describe()


# In[35]:


iqr_ = np.percentile(df.person_per_household,q=[25,75])
cut_ = 4*(iqr_[1]-iqr_[0]) + iqr_[1]
print("IQR:",cut_)


df = df[df.person_per_household < cut_] #75th Percentile oder auch 3. Quantil wird als Obergrenze für Personen pro Haushalt verwendet.
df.describe()


# In[36]:


sns.boxplot(data=df, x="person_per_household", y="priceCat")


# In[37]:


L = df.index[(df['rooms_per_household']<1)|(df['person_per_household']<1)].tolist()
df = df.drop(index=L)
df.describe()


# In[38]:


a_ = "person_per_household"
b_ = "rooms_per_household"
df["sm_PpH"] = np.log(df.loc[:,[a_]]) #Log-Transformation in extra Spalte für Statsmodels OLS
df["sm_RpH"] = np.log(df.loc[:,[b_]]) #Log-Transformation in extra Spalte für Statsmodels OLS


# In[39]:


hue_ = "priceCat"
sns.kdeplot(hue=hue_ , x="share_bedrooms",data=df)


# In[40]:


# Plots zur Kontrolle der erfolgreichen Log-Transformation
sns.jointplot(hue=hue_ , x="person_per_household", y="sm_PpH", data=df)
sns.jointplot(hue=hue_ , x="rooms_per_household", y="sm_RpH",data=df)


# ## Lage-Kategorien
# 
# ### Zusammenfassung Preiskategorie
# 
# Bisher wurden die Verteilungen mit Unterschieden bzgl. Preiskategorie betrachtet.
# Es bleibt festzuhalten, dass nur das Median-Einkommen und die Lage einen Unterschied ausmachen.
# 
# 
# ### Die Lage-Kategorie: *ocean_proximity*
# 
# Der Datensatz unterscheidet nach insgesamt fünf Lagen. Das folgende Säulendiagramm zeigt die Anzahl der Datensätze zur jeweiligen Lage, gefolgt von einem Boxplot.
# 
# 1. Weniger als 1H zum Ozean
# 2. Inland
# 3. Insel
# 4. In der Nähe zur Bucht
# 5. In der Nähe zum Ozean
# 
# Trotz der sehr geringen Anzahl von 5 Stück hat die Insel-Lage noch einen weit spannenden Boxplot, jedoch mit Median deutlich über den der anderen Kategorien.</br>
# Es ist möglich, dass diese Datensätze gelöscht oder in eine passende Kategorie überführt werden.
# 
# Die Datensätze *Nähe zur Bucht* und *Nähe zum Ozean* sind vergleichbar groß und haben eine nahezu identische Verteilung. Eine Zusammenlegung ist sinnvoll. </br>
# Damit entsteht eine gemeinsame Lage für Küstennähe: *Coast*. Fraglich ist nur ob eine Zusammenlegung mit *<1H OECAN* sinnvoll wäre. Leider stehen keine Informationen zur Datenerhebung zur Verfügung und es kann nicht geprüft werden wie stark der Unterschied zwischen Küstennähe und *<1H OCEAN* tatsächlich ist.
# 
# Folgende Argumentation steht hinter der Entscheidung aus die Lage-Kategorie auf zwei Ausprägungen zu verdichten.</br>
# Die neue Ausprägung lautet *Coast* oder *Inland* und wird in der Spalte *Proximity* angegeben. Die Werte für *Island* gehen in *Coast* ein.
# 
# **Für die Verknüpfung von *<1H Ocean* mit der neuen *Coast*-Lage sprechen:**
# * Bei der Angabe von einer Stunde könnte es sich um einen Fußweg oder eine Autofahrt handeln.
# * Alle Mediane liegen laut Boxplot über dem von *Inland*.
# * Die Verteilung für die Coast-Lagen liegen vollständig unterhalb der von *<1H OCEAN*. Siehe dazu Dichteplot unter dem Boxplot.
# * Laut Boxplot wird die Lage *Inland* mit geringeren Immobilienwerten in Verbindung gesetzt.
# * Bei der Analyse von *median_house_value* wurde eine deutliche Unterrepräsentation in der Anzahl an *below*-Werten in der Preiskategorie festgestellt.
# * Das Verhältnis von *below* zu *above* entspricht ungefähr dem Verhältnis von *Inland* zu *Coast inkl. <1H Ocean*
#     * *below/above* = 7486 / 12947 = 57.8 %
#     * *Inland/Coast inkl. <1H Ocean* = 6436 / (8473+5+2065+2395) = 49.8 %
# 
# **Dagegen sprechen:**
# * In den Lagen könnten Einkommen etc. unterschiedlich verteilt sein und entsprechende Informationen verloren gehen. **Aber diese Zusammenhänge wollen wir nicht modellieren.**

# In[41]:


op = sns.countplot(x="ocean_proximity", data=df)
for p in op.patches:
    op.annotate(f'\n AMT: {p.get_height()}', (p.get_x()+0.4,p.get_height()+950), ha="center", va="top", color="black", size="10")
plt.ylim(0,10000)
plt.show()

sns.boxplot(data=df,x="ocean_proximity",y="median_house_value")


# In[42]:


sns.kdeplot(x="median_house_value", hue="ocean_proximity", data=df)


# In[43]:


x = df.ocean_proximity
df["proximity"] = np.where((x=="NEAR BAY") | (x=="NEAR OCEAN") | (x=="<1H OCEAN") | (x=="ISLAND"),"COAST",x)
df["proximity"] = df["proximity"].astype("category")
df = df.drop("ocean_proximity", axis=1)
sns.kdeplot(x="median_house_value", hue="proximity", data=df)


# # Data Export
# 
# * Der analysierte Datensatz enthält weiterhin zukünftige Test- und Trainingsdaten, wie in Introduction besprochen.
# * Die durchgeführten Transformationen haben keine Informationen von Mittelwerten oder sonstigen Lagemaßen verwendet.
#     * D.h. diese werden nicht in der nachfolgenden Pipeline oder bei getrennter Modellierung beachtet werden.
# * 1266 Datensätze wurden entfernt.
# 
# Nachfolgend eine Übersicht zum finalen Datensatz und eine Export-Codezeile.

# In[44]:


df.info()


# In[45]:


df.to_csv("ready_data.csv", index=False)

