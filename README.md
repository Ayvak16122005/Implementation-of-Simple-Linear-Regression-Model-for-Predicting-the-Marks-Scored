# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:Kavya T 
RegisterNumber:2305003004  
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
```
Dataset:
```
![Screenshot 2024-03-19 223350](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/ebd25573-8fb1-4c78-8931-84ef5a02e4b3)
```
Head values:
```
![Screenshot 2024-03-19 223444](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/8baba54e-69d6-463b-8d28-f947e48d1152)
```
Tail values:
```
![Screenshot 2024-03-19 223611](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/f6076982-55bd-47fe-99a4-4a0a3f001736)
```
X and Y values:
```
![Screenshot 2024-03-19 223948](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/8a33584f-4a62-4cdd-b8c3-7a2b5cf54910)
```
Predication values of X and Y:
```
![Screenshot 2024-03-19 224052](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/b9d951c3-6f2d-4c41-a932-5de1edafe7b6)
```
Training Set:
```
![Screenshot 2024-03-19 224142](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/028eb3ce-1dc8-40a1-978b-87601ebac8ca)
```
Testing Set:
```
![Screenshot 2024-03-19 224238](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/bfac78be-3b4e-4418-8f34-b8aa5a8b29e5)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
