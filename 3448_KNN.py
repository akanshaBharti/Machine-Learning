from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np

df = pd.read_csv('apple_quality.csv', sep=',')
print(df.head())

label_encoder = LabelEncoder()
df['Quality'] = label_encoder.fit_transform(df['Quality'])

y = df['Quality']
X = df.drop(['Quality'], axis=1)

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X,y)
y_predict=knn.predict(X)

score=metrics.accuracy_score(y, y_predict)
print(score)
print(metrics.confusion_matrix(y, y_predict))



k_range = range(3, 10)
p_range = range(1, 4)  # p values for Minkowski distance
score_list = []  

best_score = 0
best_k = 0
best_p = 0
for i in k_range:
    for p in p_range:
        knn = KNeighborsClassifier(n_neighbors=i, p=p)
        knn.fit(X, y)
        y_predict = knn.predict(X)
        score = metrics.accuracy_score(y, y_predict)
        score_list.append(score)  
        if score > best_score:
            best_score = score
            best_k = i
            best_p = p

print("Best k:", best_k)
print("Best p:", best_p)
print("Best accuracy:", best_score)

import matplotlib.pyplot as plt
plt.plot(k_range, score_list)
plt.xlabel('values for k')
plt.ylabel('score')
plt.show()
