# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:28:07 2024

@author: HP
"""
     

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import datasets

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

#Importing Csv File
df = pd.read_csv('apple_quality.csv', sep=',')

#Encoding two categorical attributes
label_encoder = LabelEncoder()
df['Quality'] = label_encoder.fit_transform(df['Quality'])

y = df['Quality']
X = df.drop(['Quality'], axis=1)

#Checking Data
print(df.head())
print(df.columns)
print(df.index)

#Checking Null Values in Columns
print(df.isnull().sum())

correlationsall = df.corr()
print(correlationsall)


# Pearson correlation of each variable with the target
correlations = df.corr()['Quality'].abs()
correlations = correlations.sort_values(ascending=False)


selected_features = correlations.index[1:]
print(selected_features)


results_list = []

for k in range(1, len(selected_features) + 1):
    feature_subset = selected_features[:k]
    
    X_subset = df[feature_subset]
    
    X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, stratify=y, random_state=2)
    
    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    results_list.append({'Features': str(feature_subset), 'Accuracy': accuracy*100})

    print(f"Features: {feature_subset}, Accuracy: {accuracy*100:.2f}%")

results_df = pd.DataFrame(results_list)
print(results_df)
results_df.to_excel('appleQualityResults.xlsx', index=False)

