# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 00:04:19 2019

@author: Cho Yi Ru
"""


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report ,accuracy_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score


df = pd.read_csv("winequality-red.csv")

#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'fixed acidity', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'volatile acidity', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'citric acid', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'residual sugar', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'chlorides', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'sulphates', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'alcohol', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'density', data = df)
#fig = plt.figure(figsize = (10,6))
#sns.barplot(x = 'quality', y = 'pH', data = df)
#fig = plt.figure(figsize = (10,6))

df['quality'].value_counts().plot.bar()
plt.show()

df['quality'] = df['quality'].map({
        3 : 0,
        4 : 0,
        5 : 0,
        6 : 0,
        7 : 1,
        8 : 1         
})


y = df['quality']
df = df.drop('quality', 1)
X_train, X_test, Y_train, Y_test = train_test_split(
    df,
    y, 
    test_size = 0.2,
    random_state=4
)
decision_tree = DecisionTreeClassifier(random_state=4)
decision_tree.fit(X_train, Y_train)

scores = cross_val_score(decision_tree, X_train, Y_train, cv=5)
Sum = 0 
for i in range(len(scores)):
    Sum += scores[i]
Valid = Sum/5
train_pred = decision_tree.predict(X_train)

y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
label = set()
for y_p in y_pred:
    label.add(y_p)
label = list(label)
print(classification_report(Y_test , y_pred , labels=label))
print("Training accuracy: ", accuracy_score(Y_train, train_pred))
print("Validation accuracy: ", Valid)
print("Accuracy: ",accuracy_score(Y_test, y_pred))
print("MSE: " , mean_squared_error(Y_test,y_pred))
print("RMSE: " , np.sqrt(mean_squared_error(Y_test,y_pred)))
print("MAE: " , mean_absolute_error(Y_test,y_pred))
print("R_square: " , r2_score(Y_test,y_pred))




