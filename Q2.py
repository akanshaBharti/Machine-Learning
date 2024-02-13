# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:04:34 2024

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
df = pd.read_csv('apple_quality.csv')
#Encoding two categorical attributes
label_encoder = LabelEncoder()
df['Quality'] = label_encoder.fit_transform(df['Quality'])


y = df['Quality']
X = df.drop(['Quality'], axis=1)

print("Checking data")
#Checking Data
print(df.head())
print(df.columns)
print(df.index)

print("checking null values in columns")
#Checking Null Values in Columns
print(df.isnull().sum())

# Drop rows with null values
df.dropna(inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)

print("checking null values in columns")
print(df.isnull().sum())

print("Correlation Matrix:")
correlation_matrix = df.corr()
print(correlation_matrix)
# We have seen that the attributes are loosly correlated. This means that even if there is some correlation between features, 
#GNB can still perform well as long as the correlation is not strong.
# Calculate the number of samples for each class
class_counts = df['Quality'].value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,stratify=y, random_state=1)
gnb = GaussianNB()
model = gnb.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Number of wrong predictions out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))
 

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


number_of_splits = 10
accuracies = []


for i in range(number_of_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,stratify=y)
    gnb = GaussianNB()
    model = gnb.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    
    print("Accuracy for ",i + 1 , " split : ", accuracy*100)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("~" * 40)

average_accuracy = np.mean(accuracies)
print("Accuracy for ",number_of_splits, " splits : ", average_accuracy*100)

