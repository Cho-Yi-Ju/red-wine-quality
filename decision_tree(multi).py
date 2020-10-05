# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 16:30:04 2019

@author: Cho Yi Ru
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import classification_report ,accuracy_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score


data = pd.read_csv("winequality-red.csv", header = 0 , encoding="utf-8")
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'fixed acidity', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'volatile acidity', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'citric acid', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'residual sugar', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'chlorides', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'sulphates', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'alcohol', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'density', data = data)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'pH', data = data)
#fig = plt.figure(figsize = (10,6))
feature_name = ["fixed acidity" , "volatile acidity","citric acid","residual sugar","chlorides"
                ,"free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

X = data[feature_name]
y = data.quality

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


Decision_tree = DecisionTreeClassifier(criterion="gini", max_depth=100 , random_state=10)

decision = Decision_tree.fit(X_train , y_train)
scores = cross_val_score(decision, X_train, y_train, cv=5)

Sum = 0 
for i in range(len(scores)):
    Sum += scores[i]
Valid = Sum/5
train_pred = decision.predict(X_train)
y_pred = decision.predict(X_test)
label = set()
for y_p in y_pred:
    label.add(y_p)
label = list(label)

print(classification_report(y_test , y_pred , labels=label))
print("Training accuracy: ", accuracy_score(y_train, train_pred))
print("Validation accuracy: ", Valid)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("RMSE: " , np.sqrt(mean_squared_error(y_test,y_pred)))
print("MSE: " , mean_squared_error(y_test,y_pred))
print("MAE: " , mean_absolute_error(y_test,y_pred))
print("R_square: " , r2_score(y_test,y_pred))


