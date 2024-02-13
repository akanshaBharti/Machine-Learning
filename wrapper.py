from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

L= load_iris()
X, y = load_iris(return_X_y=True)

# Round 1

total_accuracies = []
df = pd.DataFrame(X)
feature=[0,1,2,3]
accuracies1 = []
for i in feature:
    X_train, X_test, y_train, y_test = train_test_split(X[:,i], y, test_size=0.4, stratify=y)
    gnb = GaussianNB()
    model= gnb.fit(X_train.reshape(-1, 1), y_train)
    y_pred=gnb.predict(X_test.reshape(-1, 1))
    accuracy=accuracy_score(y_test, y_pred)
    accuracies1.append(accuracy)
    print(f"Accuracy for feature {i}: {accuracy}")

max_value=np.argmax(accuracies1)
print(max_value)
selectedFeature1= feature[max_value]
# print(selectedFeature1)
feature.pop(max_value);
print(feature)
total_accuracies.append(max(accuracies1))

#Round 2
accuracies2 = []
for i in feature:
    selected = [selectedFeature1] + [i]
    
    X_train, X_test, y_train, y_test = train_test_split(df[selected], y, test_size=0.4, stratify=y)
    gnb = GaussianNB()
    model= gnb.fit(X_train, y_train)
    y_pred=gnb.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    accuracies2.append(accuracy)
    print(f"Accuracy for feature {max_value,i}: {accuracy}")

max_value2=np.argmax(accuracies2)
print("maxvalueindex",max_value2)
selectedFeature2= [selectedFeature1] + [feature[max_value2]]
print("selected feature",selectedFeature2)
feature.pop(max_value2);
print(feature)

total_accuracies.append(max(accuracies2))

#round 3
accuracies3 = []
for i in feature:
    selected = selectedFeature2 + [i]
    
    X_train, X_test, y_train, y_test = train_test_split(df[selected], y, test_size=0.4, stratify=y)
    gnb = GaussianNB()
    model= gnb.fit(X_train, y_train)
    y_pred=gnb.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    accuracies3.append(accuracy)
    print(f"Accuracy for feature {max_value2,i}: {accuracy}")

max_value3=np.argmax(accuracies3)
print("maxvalueindex",max_value3)
selectedFeature3= selectedFeature2 + [feature[max_value3]]
print("selected feature",selectedFeature3)
feature.pop(max_value3);
print(feature)

total_accuracies.append(max(accuracies3))
#round 4
accuracies4 = []
for i in feature:
    selected = selectedFeature3 + [i]
    
    X_train, X_test, y_train, y_test = train_test_split(df[selected], y, test_size=0.4, stratify=y)
    gnb = GaussianNB()
    model= gnb.fit(X_train, y_train)
    y_pred=gnb.predict(X_test)
    accuracy=accuracy_score(y_test, y_pred)
    accuracies4.append(accuracy)
    print(f"Accuracy for feature {max_value2,i}: {accuracy}")

max_value4=np.argmax(accuracies4)
print("maxvalueindex",max_value4)
selectedFeature4= selectedFeature3 + [feature[max_value4]]
print("selected feature",selectedFeature4)
feature.pop(max_value4);
print(feature)

total_accuracies.append(max(accuracies4))


dataframe1 = pd.DataFrame({
    "Round": range(1, 5),
    "Selected Features": [selectedFeature1, selectedFeature2, selectedFeature3, selectedFeature4],
    "Accuracy": total_accuracies
})
    
print(dataframe1)







    

    
    




