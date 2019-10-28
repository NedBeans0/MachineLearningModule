# import all the packages needed for this workshop

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import warnings
import math
warnings.filterwarnings("ignore")
# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
'''
#Load dataset and show data frame
'''
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston

Dibs = load_diabetes()
X = Dibs.data
Y = Dibs.target
df = pd.DataFrame(X)
df.columns = Dibs.feature_names
print(df.head())


#Divide dataset into  a training set and a test set, the test set size is 20% of the total data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2)

#Create linear regressor, train the model, then make predictions for the test data
algo = LinearRegression()
algo.fit(X_train,Y_train)
Y_pred = algo.predict(X_test)

'''
Evaluate the model on the train set and test set, respectively

Note: algo.score returns R squared value
'''
s1=algo.score(X_train,Y_train)
s2=algo.score(X_test, Y_test)
print(s1, s2)

'''
Now do 5-fold cross validation using the cross_val_score function. 
Think about which performance score you should use. 
Go to https://scikit-learn.org/stable/modules/model_evaluation.html and see alternative scoring parameters.
'''

# Example using r-squared metric as learned in the statquest video
print("The mean of r2: ")
print(cross_val_score(algo, X, Y, scoring='r2', cv = 5).mean())

#Visualize the test set and the prediction in  a scatter plot
plt.scatter(Y_pred,Y_test, color='black')
plt.plot(Y_test, Y_test,  color='blue', linewidth=3)
plt.show()

'''
Visualise absolute error by boxplot and density plot, reasons for doing so on lecture slides
'''
# box plot
abs_error=abs(Y_test-Y_pred)

df = pd.DataFrame({'error':abs_error})
df.boxplot(grid=True)

# density plot
sn.kdeplot(df['error'], shade=True)









'''
EXERCISES
'''
# load boston housing pricing dataset
boston=load_boston()
boston_df=pd.DataFrame(boston.data,columns=boston.feature_names)

# display basic information of boston dataset
print(boston.DESCR)
print(boston_df.info())
boston_df['Price']=boston.target

input_data=boston_df.drop('Price',axis=1)

output_data=boston.target
# Create a linear regressor and a ridge regressor:

lin= LinearRegression()

rdg= Ridge()

# Create the test train split (2/3 train data, 1/3 test data), using train_test_split
BX_train, BX_test, BY_train, BY_test = train_test_split(input_data, output_data, test_size = 0.33)

# Run the data with linear regression, return R squared value
lin.fit(BX_train, BY_train)
BY_pred = lin.predict(BX_test)

Bs1=lin.score(BX_train,BY_train)
Bs2=lin.score(BX_test, BY_test)
print(Bs1, Bs2)

# Run the data with ridge Regression, return R squared value
rdg.fit(BX_train, BY_train)
ridgeBY_pred = rdg.predict(BX_test)

ridgeBs1=rdg.score(BX_train,BY_train)
ridgeBs2=rdg.score(BX_test, BY_test)
print(ridgeBs1, ridgeBs2)


# put your code here to calculate absolute errors for linear and ridge regression

print("LIN: ")
print('Mean Absolute Error:')
print(mean_absolute_error(BY_test, BY_pred))
print('Mean Squared Error:',  )
print(mean_squared_error(BY_test, BY_pred))
print('Root Mean Squared Error:',  )
print(math.sqrt(mean_squared_error(BY_test, BY_pred)))

print("\nRIDGE: ")
print('Mean Absolute Error:')  
print(mean_absolute_error(BY_test, ridgeBY_pred))
print('Mean Squared Error:',  )  
print(mean_squared_error(BY_test, ridgeBY_pred))
print('Root Mean Squared Error:')
print(math.sqrt(mean_squared_error(BY_test, ridgeBY_pred)))

# Print the mean of r2 scores for the 2 algorithms, using the cross_val_score function
print("LIN: ")
cross_val_score(Bs2, ridgeBs2)


'''
TO DO LAST QUESTION...
'''
