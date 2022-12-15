# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 10:06:30 2022

@author: admin
"""

import pandas as pd
df = pd.read_csv("delivery_time.csv")
df.shape
df.head()
#=========================================================    

# step2: split the Variables in  X and Y's
X = df[["Delivery Time"]]
Y = df[["Sorting Time"]]

X.ndim

#=========================================================    
# EDA
# scatter plot
import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,0], Y,color='red')
plt.ylabel("Sorting Time")
plt.xlabel("Delivery Time")
plt.show()

#### Histogram and Density Plots
# create a histogram plot
from matplotlib import pyplot
df.hist()
pyplot.show()

df.skew()
df.describe()
df.kurtosis()


import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.rand(10, 2), columns=['delivery time','Sorting time'])
df.plot.box(grid='True')
 
# Data transformation                    
# scaling
from sklearn.preprocessing import MinMaxScaler
MM = MinMaxScaler()
MM_X = pd.DataFrame(MM.fit_transform(X))

# data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

#=========================================================    
# Model fitting  --> Scikit learn
from sklearn.linear_model import LinearRegression
LR = LinearRegression()

LR.fit(X,Y)

LR.intercept_ 
LR.coef_ # B1

#=========================================================    

Y_pred = LR.predict(X)
Y_pred

Y_pred_train = LR.predict(X_train) 
Y_pred_test = LR.predict(X_test) 


from sklearn.metrics import mean_squared_error
mse1= mean_squared_error(Y_train,Y_pred_train)
RMSE1 = np.sqrt(mse1)
print("Training error: ", RMSE1.round(2))

mse2= mean_squared_error(Y_test,Y_pred_test)
RMSE2 = np.sqrt(mse2)
print("Test error: ", RMSE2.round(2))
from sklearn.metrics import mean_squared_error, r2_score
mse= mean_squared_error(Y,Y_pred)

import numpy as np
RMSE = np.sqrt(mse)
print("Root mean squarred error: ", RMSE.round(2))
print("Rsquare: ",r2_score(Y,Y_pred).round(3)*100)
