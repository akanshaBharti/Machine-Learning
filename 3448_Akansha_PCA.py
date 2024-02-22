
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

df = pd.read_csv('apple_quality.csv', sep=',')
print(df.head())

scaler = StandardScaler()

#Encoding categorical attributes
label_encoder = LabelEncoder()
df['Quality'] = label_encoder.fit_transform(df['Quality'])

y = df['Quality']
X = df.drop(['Quality'], axis=1)

#Checking Data
print(df.head())
print(df.columns)
print(df.index)

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA()
X_new=pca.fit_transform(X)
gnb = GaussianNB()

result = pd.DataFrame(columns=['Features', 'Info', 'accuracy'])
for i in range(0,X_new.shape[1]):
    X_train, X_test, y_train, y_test = train_test_split(X_new[:,:i+1], y, test_size=0.4, random_state=0)
    model = gnb.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)*100
    sum_info = sum(pca.explained_variance_ratio_[:i+1])
    print(f"Sum info k{i+1} : {sum_info}")
    print(f"Accuracy k{i+1} : {accuracy}")
    result.loc[len(result.index)] = [i+1, sum_info, accuracy]

result.to_excel('3448_PCA.xlsx', index=False)


