# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:05:39 2019

@author: Cho Yi Ru
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import classification_report ,accuracy_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score


data = pd.read_csv("winequality-red.csv", header = 0 , encoding="utf-8")

feature_name = ["fixed acidity" , "volatile acidity","citric acid","residual sugar","chlorides"
                ,"free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

X = data[feature_name]
y = data.quality

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


RandomForest = RandomForestClassifier(n_estimators = 100 , random_state=4)
decision = RandomForest.fit(X_train , y_train)
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

#mat = confusion_matrix(y_test, y_pred)
#sns.heatmap(mat.T,xticklabels = True, yticklabels=True, square=True, annot=True, fmt='d', cbar=True)
#plt.xlabel('true label')
#plt.ylabel('predicted label')
print(classification_report(y_test , y_pred , labels=label))
print("Training accuracy: ", accuracy_score(y_train, train_pred))
print("Validation accuracy: ", Valid)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print("RMSE: " , np.sqrt(mean_squared_error(y_test,y_pred)))
print("MSE: " , mean_squared_error(y_test,y_pred))
print("MAE: " , mean_absolute_error(y_test,y_pred))
print("R_square: " , r2_score(y_test,y_pred))


