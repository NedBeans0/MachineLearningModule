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

    For each variable in x, a row : [1, x, x², x³... xⁿ] where n based on the degrees
    '''
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    

    return X

def getWeightsForPolynomialFit(x,y,degree):
    #Least squares estimation to get the weights for the equation
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y))

    return w


def getEquation(x, weights):
    #Plugs value into the polynomial equation
    yintercept = weights[0]
    print('Y Intercept = ');print(yintercept)

    for i in np.arange(1, len(weights)):            
        yintercept += weights[i] * x ** i 
    return yintercept


def pol_regression(features_train, y_train, degree):
    #Calls function to use least squares solution to get weights
    weights = getWeightsForPolynomialFit(features_train, y_train, degree)  
    line = getEquation(features_train, weights)

    plt.plot(features_train, line)        
    plt.title('Polynomial Fit')

    plt.xlabel('x')
    plt.ylabel('y') 
    #plt.show()
    return weights






def eval_pol_regression(x, y):
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
    polydata1 = pd.read_csv('pol_regression.csv')
    poly2 = polydata1.sample(frac=1)

    train_data = poly2[0:(int(round(len(poly2)*0.7)))]
    test_data = poly2[(int(round(len(poly2)*0.7))):(len(poly2))]

    x_train = train_data['x'].values
    y_train = train_data['y'].values

    x_test = test_data['x'].values
    y_test = test_data['y'].values

    print('X and Y Train')
    print(x_train)
    print(y_train)

    print('X and Y Test')
    print(x_test)
    print(y_test)
    rmse_train = np.zeros((9,1))
    rmse_test = np.zeros((9,1))

    for i in range(1,10):
        Xtrain2 = getPolynomialDataMatrix(x_train,i)
        Xtest2 = getPolynomialDataMatrix(x_test,i)
        if i>=1:
            w = getWeightsForPolynomialFit(x_train,y_train,i)
        elif i == 0:
            w = np.mean(y_train)
        rmse_train[i-1] = np.sqrt(np.mean((Xtrain2.dot(w)-y_train)**2))
        rmse_test[i-1] = np.sqrt(np.mean((Xtest2.dot(w)-y_test)**2))

    plt.semilogy(range(1,10),rmse_train)
    plt.semilogy(range(1,10),rmse_test)
    plt.legend(('RMSE Training', 'RMSE Testing'))
    plt.show()





    #rmse = 
    #return rmse



plt.plot(x_train, y_train, 'bo') 
weights1 = pol_regression(x_train, y_train, 1)

line2 = pol_regression(x_train, y_train, 2)
line3 = pol_regression(x_train, y_train, 3)
line4 = pol_regression(x_train, y_train, 5)
line5 = pol_regression(x_train, y_train, 10)

#np.mean(y_train) and plot like ususal 

plt.legend(('points', '$x$', '$x^2$', '$x^3$', '$x^5$','$x^{10}$'), loc = 'lower right')
plt.show()

eval_pol_regression(x_train, y_train)

#eval_pol_regression(weights1, x_train, y_train, 1)