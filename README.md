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
#Dataset:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/ML student_scores.csv')
print(df)
#Head values:
df.head(0)
print(df.head())
#Tail values:
df.tail(0)
print(df.tail())
#X and Y values:
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
#Predication values of X and Y:
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
 #MSE,RMSE:
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
![Screenshot 2024-03-21 214817](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/5bec79f3-3d15-496c-adfa-bd4c2a07f2d7)

```
Head values:
```
![Screenshot 2024-03-21 215106](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/3ac75b54-2758-40c8-b019-09c128d09e98)

```
Tail values:
```
![Screenshot 2024-03-21 215142](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/9a117896-0a66-4b52-bb55-c21e19f73e81)

```
X and Y values:
```
![Screenshot 2024-03-21 215259](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/c7788a9f-15a9-409e-8608-f91cecc677f2)
```
Predication values of X and Y:
```
![Screenshot 2024-03-21 215425](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/ba0f080e-ae0f-4aa0-b01d-afde530bf1fb)

```
Training Set:
```
![Screenshot 2024-03-21 215529](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/be049678-ef84-4309-b1bd-9ccd89068143)

```
Testing Set:
```
![Screenshot 2024-03-21 215604](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/593c2e79-3d1c-49a8-aee3-15f915cf74b7)
```
 MSE,RMSE:
```
![Screenshot 2024-03-21 215856](https://github.com/Ayvak16122005/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/147690197/0795ae84-072b-45ee-b92f-83f0827cfbeb)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
