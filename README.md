# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection and Preprocessing
2. Model Training
3. Model Evaluation
4. Model Deployment and Monitoring

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ANU VARSHINI M B
RegisterNumber: 212223240010
*/
```
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Employee (1).csv")
data.head()
```
```
data.info()
```
```
data.isnull()
```
```
data.isnull().sum()
```
```
data['left'].value_counts()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
```
```
data['salary']=le.fit_transform(data['salary'])
data.head()
```
```
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()
```
```
y=data['left']
y.head()
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy
```
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![alt text](<Screenshot 2024-04-06 113139.png>)

![alt text](<Screenshot 2024-04-06 113200.png>)

![alt text](<Screenshot 2024-04-06 113213.png>)

![alt text](<Screenshot 2024-04-06 113226.png>)

![alt text](<Screenshot 2024-04-06 113305.png>)

![alt text](<Screenshot 2024-04-06 113318.png>)

![alt text](<Screenshot 2024-04-06 113331.png>)

![alt text](<Screenshot 2024-04-06 113339.png>)

![alt text](<Screenshot 2024-04-06 113347.png>)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
