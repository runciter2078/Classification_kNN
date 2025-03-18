#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KNN Classifier with MinMax Normalization for SPY Data

This script loads a CSV dataset ("SPYV3.csv"), selects a subset of features,
normalizes the data using MinMaxScaler, and then performs hyperparameter tuning
using RandomizedSearchCV for a k-NN classifier. Finally, it trains the classifier
with the selected parameters, evaluates the model via classification reports and a 
confusion matrix, and prints the results.

Author: Pablo Beret
Created on Mon Sep  3 19:21:48 2018 (updated version)
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, make_scorer, classification_report
import warnings

warnings.filterwarnings('ignore')

# Load dataset with selected features (excluding the classifier column for normalization)
data = pd.read_csv("SPYV3.csv", sep=',', usecols=['1', '42', '45', '47', '60',
                                                   '73', '171', '179', '187', '221', 'FECHA.month'])
# Load the classifier column separately
clasificador = pd.read_csv("SPYV3.csv", sep=',', usecols=['CLASIFICADOR'])

# Normalize the dataset (excluding the classifier)
min_max_scaler = preprocessing.MinMaxScaler()
data_normalized = min_max_scaler.fit_transform(data)
data_normalized = pd.DataFrame(data_normalized, columns=['1', '42', '45', '47', '60', '73',
                                                         '171', '179', '187', '221', 'FECHA.month'])

# Add back the classifier column
data_normalized['CLASIFICADOR'] = clasificador

# Order the columns so that 'CLASIFICADOR' comes first
data_normalized = data_normalized[['CLASIFICADOR','1','42','45','47','60','73',
                                   '171','179','187','221','FECHA.month']]

# Free memory (remove the temporary variable)
del(clasificador)

# Split dataset into training and testing sets
p_train = 0.75  # Percentage for training set
train = data_normalized[:int(len(data_normalized) * p_train)]
test = data_normalized[int(len(data_normalized) * p_train):]

print("Training examples:", len(train))
print("Testing examples:", len(test))
print("\n")

# Define features and target variable
features = data_normalized.columns[1:]
x_train = train[features]
y_train = train['CLASIFICADOR']

x_test = test[features]
y_test = test['CLASIFICADOR']

# Hyperparameter tuning using RandomizedSearchCV
from scipy.stats import randint as sp_randint

clf = KNeighborsClassifier()  # Build the classifier

# Build the scorer using precision_score (binary average)
metrica = make_scorer(precision_score, pos_label=1, greater_is_better=True, average="binary")

def report(results, n_top=1):
    """
    Report the top n models from the hyperparameter search.
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {}".format(i))
            print("Mean validation score: {:.3f} (std: {:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {}".format(results['params'][candidate]))
            print("")

# Parameters and distributions for sampling
param_dist = {
    'n_jobs': [-1],
    'n_neighbors': [7,8,9,10,11,12,13],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    'leaf_size': [40, 45, 50],
    'metric': ['chebyshev']
}

n_iter_search = 168
random_search = RandomizedSearchCV(clf, scoring=metrica, 
                                   param_distributions=param_dist, 
                                   n_iter=n_iter_search)
random_search.fit(x_train, y_train)
report(random_search.cv_results_)

# Build the final k-NN classifier with the chosen parameters (example values)
final_clf = KNeighborsClassifier(
    weights='uniform',
    n_neighbors=11,
    n_jobs=-1,
    metric='chebyshev',
    leaf_size=45,
    algorithm='kd_tree'
)

final_clf.fit(x_train, y_train)  # Train the model

preds = final_clf.predict(x_test)  # Test the model

# Evaluate the model
from sklearn.metrics import classification_report
print("k-NN Classifier Results:\n" + classification_report(y_true=test['CLASIFICADOR'], y_pred=preds))

# Display confusion matrix
print("Confusion Matrix:\n")
confusion = pd.crosstab(test['CLASIFICADOR'], preds, rownames=['Actual'], colnames=['Predicted'])
print(confusion)
