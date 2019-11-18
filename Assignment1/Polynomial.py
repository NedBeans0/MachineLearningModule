import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import linalg
from collections import OrderedDict

'''
pol_regression.csv is a one-dimensional dataset with inputs x (1 input dimension) and outputs y (1 output dimension).


'''

data_train = pd.DataFrame.from_csv('pol_regression.csv')
data_train.sort_values('x', axis = 0, ascending = True, inplace = True, na_position = 'last')
x_train = data_train['x'].as_matrix()
y_train = data_train['y'].as_matrix()



def getPolynomialDataMatrix(x, degree):
    '''
    This function does exactly the same as np.vander()
    except np.vander() arranges the numbers in the opposite 
    order. Also this runs slightly quicker.
    Essentially arranges a Vandermonde matrix with a row
    for each variable and the amount of columns (and therefore
    the amount of incrementing exponents) is based
    on the degree. 

    For each variable a row : [1, x, x², x³... xⁿ] where n based on the degrees
    '''
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))

    return X

def getWeightsForPolynomialFit(x,y,degree):
    #Least squares estimation to get the weights for the equation
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y_train))

    return w





def pol_regression(features_train, y_train, degree):
    #Creates a vandermonde matrix from the x column which essentially gets us the coefficients in the equation
    coefficients = getPolynomialDataMatrix(features_train, degree)
    print('Coefficients:');print(coefficients)
    #Calls function to use least squares solution to get weights
    
    #Delete this eventually, it now does the same as theta
    # theta2 = np.matmul(np.matmul(linalg.pinv(np.matmul(np.transpose(coefficients),coefficients)), np.transpose(coefficients)), y_train)



    weights = getWeightsForPolynomialFit(features_train, y_train, 4)
    
    print('Theta Weight Values:');print(weights)
    plt.figure()
    plt.plot(x_train, y_train, 'bo')

    yintercept = weights[0]
    print('Y Intercept = ');print(yintercept)

    for i in np.arange(1, len(weights)):            
        yintercept += weights[i] * features_train ** i 


    plt.plot(features_train, yintercept)        
    plt.title('Polynomial Fit: Order ' + str(len(weights)-1))

    plt.xlabel('x')
    plt.ylabel('y') 
    plt.show()



   





    
    '''
    plt.figure()
    #plt.plot(x_test,y_test, 'g')
    plt.plot(features_train,y_train, 'bo')
    plt.show()
    w4 = getWeightsForPolynomialFit(x_train,y_train,4)
    Xtest4 =getPolynomialDataMatrix(x_train, 4)
    ytest4 = Xtest4.dot(w4)
    plt.plot(Xtest4, ytest4, 'r')
    plt.show()
    '''

    
















    

    #parameters = 
    ##note - parameters should be 1D numpy array of size n+1 where n is the degree of the polynomial function
    #return parameters
 
pol_regression(x_train, y_train, 4)

def eval_pol_regression(parameters, x, y, degree):
    '''
The function takes the parameters computed by the pol_regression function and evaluates its
performance on the dataset given by the input values x and output values y (again 1D numpy arrays).
The last argument of the function again specifies the degree of the polynomial. In this function, you
need to compute the root mean squared error (rmse) of the polynomial given by the parameters
vector. 

Now you again need to train your polynomial functions with degrees 0, 1, 2, 3, 5 and 10. However,
this time, split the dataset provided (same as before) into 70% train and 30% test set. Evaluate the
training set rmse and the test set rmse of all the given degrees. Plot both rmse values using the degree
of the polynomial as x-axis of the plot. Interpret your results. Which degree would you now choose?
Are there any degrees of the polynomials where you can clearly identify over and underfitting?
Explain your conclusions! 
    '''
    print('stop errror')
    #rmse = 
    #return rmse