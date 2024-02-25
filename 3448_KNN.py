from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
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
print("Score:", score)
print("Confusion Matrix:\n", metrics.confusion_matrix(y, y_predict))



k_range = range(3, 10)
p_range = range(1, 4)  # p values for Minkowski distance
weights = ['uniform', 'distance']
score_list = []  

best_score = 0
best_k = 0
best_p = 0
best_weights = ''
for i in k_range:
    for p in p_range:
        for weight in weights:
            knn = KNeighborsClassifier(n_neighbors=i, p=p, weights=weight)
            knn.fit(X, y)
            y_predict = knn.predict(X)
            score = metrics.accuracy_score(y, y_predict) 
            if score > best_score:
                best_score = score
                best_k = i
                best_p = p
                best_weights = weight
                score_list.append(score)


print("Best k:", best_k)
print("Best p:", best_p)
print("Best accuracy:", best_score)
print("Best weights:", best_weights)
print("Score_list",score_list)

cv_scr=cross_val_score(knn, X, y, cv=5) # cv=5 crossVal on 5 folds
print("Crossval:", cv_scr)
print("CrossVal mean:", np.mean(cv_scr))
print("crossVal std:", np.std(cv_scr))

y_pred = cross_val_predict(knn, X, y, cv=5)
accuracy = metrics.accuracy_score(y, y_pred)
precision = metrics.precision_score(y, y_pred, average='weighted')
recall = metrics.recall_score(y, y_pred, average='weighted')
f1 = metrics.f1_score(y, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

