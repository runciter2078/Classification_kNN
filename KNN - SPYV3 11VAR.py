# -*- coding: utf-8 -*-
"""
KNN NORMALIZADO MINMAX

Created on Mon Sep  3 19:21:48 2018

@author: Pablo Beret
"""

# Carga de datos de un CSV con delimitador coma y selección de variables.

import pandas as pd

spy = pd.read_csv("SPYV3.csv", sep=',', usecols=['1','42','45','47','60',
                                                 '73','171','179','187','221','FECHA.month'])
                                                  
clasificador = pd.read_csv("SPYV3.csv", sep=',', usecols=['CLASIFICADOR'])                                                 

# Normalización del dataset sin el clasificador ni el mes

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
spy = min_max_scaler.fit_transform(spy)

spy = pd.DataFrame(spy, columns=['1','42','45','47','60','73',
                                 '171','179','187','221','FECHA.month'])

# Añadimos el clasificador y la fecha mes al dataset normalizado

spy['CLASIFICADOR']= clasificador

# Se ordenan las columnas

spy = spy[['CLASIFICADOR','1','42','45','47','60','73',
           '171','179','187','221','FECHA.month']]

del(clasificador)

# División del conjunto en train y test

p_train = 0.75 # Porcentaje de train. Modificar para obtener diferentes conjuntos.

train = spy[:int((len(spy))*p_train)]
test = spy[int((len(spy))*p_train):]

print("Ejemplos usados para entrenar: ", len(train))
print("Ejemplos usados para test: ", len(test))
print("\n")

features = spy.columns[1:]
x_train = train[features]
y_train = train['CLASIFICADOR']

x_test = test[features]
y_test = test['CLASIFICADOR']

# Utilización de RandomizedSearchCV para busqueda de hiperparámetros

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, make_scorer
import warnings

warnings.filterwarnings('ignore') 

X, y = x_train, y_train # Datos de entrenamiento

clf = KNeighborsClassifier() # Construcción del clasificador

#Construcción de la métrica

metrica = make_scorer(precision_score, pos_label=1, 
                      greater_is_better=True, average="binary") 
                     
def report(results, n_top=1): # Función para mostrar resultados
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Parámetros y distribuciones para muestrear
param_dist = {'n_jobs': [-1], 'n_neighbors': [7,8,9,10,11,12,13],
              'weights': ['uniform','distance'],
              'algorithm': ['ball_tree','kd_tree','brute','auto'], 
              'leaf_size': [40,45,50],
              'metric': ['chebyshev']
              }

n_iter_search = 168 # Ejecución
random_search = RandomizedSearchCV(clf, scoring= metrica, 
                                   param_distributions=param_dist, 
                                   n_iter=n_iter_search)
                                   

random_search.fit(X, y)
report(random_search.cv_results_)


# Creación del modelo Decision Tree con los parámetros obtenidos

clf = KNeighborsClassifier(weights= 'uniform', n_neighbors= 11, n_jobs= -1,
                           metric = 'chebyshev', leaf_size= 45, 
                           algorithm= 'kd_tree')
          
                                  
clf.fit(x_train, y_train) # Construcción del modelo

preds = clf.predict(x_test) # Test del modelo

# Visualización de resultados

from sklearn.metrics import classification_report
print("SVM: \n" 
      +classification_report(y_true=test['CLASIFICADOR'], y_pred=preds))

# Matriz de confusión

print("Matriz de confusión:\n")
matriz = pd.crosstab(test['CLASIFICADOR'], preds, rownames=['actual'], colnames=['preds'])
print(matriz)













