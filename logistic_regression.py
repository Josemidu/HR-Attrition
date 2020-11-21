# -*- coding: utf-8 -*-
"""
Strategic Business Analytics
ESSEC Business School (2020, November) Foundations of strategic business
analytics, retrieved October 28, 2020 from ESSEC Business School

Alejandro Miguez D.
"""
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
#%%
#Human resource dataset
HR = pd.read_csv("HR2.csv", delimiter = ",")
#%%
#Exploratory analysis
HR.info()
HR.head()
HR.tail()
report = ProfileReport(HR)
report.to_file('HR_report.html')
#Now grouping by time spent in the company to get some insights
pivot = pd.pivot_table(HR, index = ['TIC'], values = ['left'], 
                       aggfunc = [np.mean, len])
pivot['Time Spent'] = [2,3,4,5,6]
pivot.info()
#Unpacking tupples to have a single joined column
pivot.columns = ['-'.join((a, b)) for a, b in pivot.columns]
pivot.reset_index()
pivot.columns = ['Attrition rate', 'amount', 'Time spent']
pivot.info()
#Now, a scatter plot showing employees' behavior
plt.scatter(pivot['Time spent'], pivot['Attrition rate'], s = pivot['amount'], 
            color = 'b', alpha = 0.5)
plt.xlabel('Time spent')
plt.ylabel('Attrition rate')

pivot2 = pd.pivot_table(HR, index = ['S'], values = ['left'], 
                        aggfunc = [np.mean, len])
pivot2.reset_index(inplace = True)
pivot2.columns = ["_".join((a,b)) for a, b in pivot2.columns]
pivot2.columns = ['Satisfaction', 'Average attrition', 'Attrition']
pivot2.sort_values(by = ['Satisfaction'], inplace = True)
plt.scatter(pivot2['Satisfaction'], pivot2['Average attrition'], color = 'r', 
            s = pivot2['Attrition'], alpha = 0.3)
plt.xlabel('Satisfaction')
plt.ylabel('Attrition')
#%%
#Building a logistic regression model
#Spliting data
x_train, x_test, y_train, y_test = train_test_split(HR.iloc[:,0:6], HR.loc[:,'left'],
                                                    test_size = 0.6, random_state = 42)
LR = LogisticRegression()
LR.fit(x_train, y_train)
#Tunning the classification threshold for business purposes
threshold = 0.5
prediction = np.where(LR.predict_proba(x_test)[:,1] > threshold, 1, 0)
#Finally, some metrics
metrics = pd.DataFrame(data = [accuracy_score(y_test, prediction), 
                               precision_score(y_test, prediction),
                               roc_auc_score(y_test, prediction),
                               recall_score(y_test, prediction)],
                       index = ["Accuracy score", "Precision score",
                                "ROC AUC", "Recall score"])
print(metrics)
cm = confusion_matrix(y_test, prediction)
print(cm)
likely_to_stay = 0
likely_to_leave = 0
for n in prediction:
    if n == 0:
        likely_to_stay += 1
    else:
        likely_to_leave += 1
        
print(likely_to_stay/len(prediction))
