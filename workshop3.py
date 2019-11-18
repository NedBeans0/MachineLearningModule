import numpy as np
import numpy.linalg as linalg
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

data_train = pd.DataFrame.from_csv('/Users/admin/Downloads/regression_train(1).csv')
data_test = pd.DataFrame.from_csv('/Users/admin/Downloads/regression_test(1).csv')

print(data_train)

x_train = data_train['x'].as_matrix()
y_train = data_train['y'].as_matrix()

x_test = data_test['x'].as_matrix()
y_test = data_test['y'].as_matrix()

plt.clf()
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.legend(('training points', 'ground truth'))
plt.savefig('trainingdata.png')
#plt.show()

#As a first step, lets construct the X tilde matrix. (the training data matrix) 
X = np.column_stack((np.ones(x_train.shape), x_train))
print('X:')
print(X)

#Now we need to construct a vector that contains the derivative (i.e. calculus stuff) of each variable in the training
#We need the derivative of E(x). Then:
# Plot  𝐸(𝑥)  and  ∂𝐸(𝑥)∂𝑥  as 3D plot. For  𝑥0  and  𝑥1 , use an interval of  [−5,5]  for the plot using  
# 51  partitions for each dimension. Confirm your finding of the minimum in the plot.
A = np.array([[1, 0.5], [0.5, 1]])
a = np.array([[1], [0]])

# specify data points for x0 and x1 (from - 5 to 5, using 51 uniformly distributed points)
x0Array = np.linspace(-5, 5, 51)
XX = X.transpose().dot(X)

w = np.linalg.solve(XX, X.transpose().dot(y_train))
#w = np.linalg.inv(XX).dot(X.transpose().dot(y_train))


#Linear regression stuff without hand picked parameters
Xtest = np.column_stack((np.ones(x_test.shape), x_test))
ytest_predicted = Xtest.dot(w)

plt.figure()
plt.plot(x_test,y_test, 'g')
plt.plot(x_test, ytest_predicted, 'r')
plt.plot(x_train,y_train, 'bo')
plt.legend(('training points', 'ground truth', 'prediction'), loc = 'lower right')
#plt.hold(True)
plt.savefig('regression_LSS.png')
#plt.show()

#Get Sum of Squared Errors
error = X.dot(w)
SSE = error.dot(error)
print('Sum of Squared Errors')
print(SSE)


'''

                P O L Y N O M I A L   R E G R E S S I O N
'''
#Instead of fitting a line, we can also fit a polynomial.

#In python, we write a small function that does the feature expansion up to a certain degree for a given data set x
def getPolynomialDataMatrix(x, degree):
    X = np.ones(x.shape)
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i))
    print('X from polynomial')
    print(X)
    return X
    
print(getPolynomialDataMatrix(x_train, 4))
#We now want to test different polynomials and see which one fits our data best.
#First, implement a function that computes the optimal beta values given the input data x, output data y 
# and the desired degree of the polynomial. 

def getWeightsForPolynomialFit(x,y,degree):
    X = getPolynomialDataMatrix(x, degree)

    XX = X.transpose().dot(X)
    w = np.linalg.solve(XX, X.transpose().dot(y_train))

    return w

#Trying out 4 different degrees 

plt.figure()
plt.plot(x_test,y_test, 'g')
plt.plot(x_train,y_train, 'bo')

w1 = getWeightsForPolynomialFit(x_train,y_train,1)
Xtest1 =getPolynomialDataMatrix(x_test, 1)
ytest1 = Xtest1.dot(w1)
plt.plot(x_test, ytest1, 'r')

w2 = getWeightsForPolynomialFit(x_train,y_train,2)
Xtest2 =getPolynomialDataMatrix(x_test, 2)
ytest2 = Xtest2.dot(w2)

plt.plot(x_test, ytest2, 'g')

w3 = getWeightsForPolynomialFit(x_train,y_train,3)
Xtest3 =getPolynomialDataMatrix(x_test, 3)
ytest3 = Xtest3.dot(w3)
plt.plot(x_test, ytest3, 'm')


w4 = getWeightsForPolynomialFit(x_train,y_train,4)
Xtest4 =getPolynomialDataMatrix(x_test, 4)
ytest4 = Xtest4.dot(w4)
plt.plot(x_test, ytest4, 'c')

plt.legend(('training points', 'ground truth', '$x$', '$x^2$', '$x^3$', '$x^4$'), loc = 'lower right')

plt.savefig('polynomial.png')
plt.show()

'''
As we can see, the predictions for 3rd and 4th order are already quite good. Lets see what happens if we increase 
the order of the polynomial. Repeat the same plots for for example 7, 10 and 12th order polynomials.
'''
plt.figure()
plt.plot(x_test,y_test, 'g')
plt.plot(x_train,y_train, 'bo')

w7 = getWeightsForPolynomialFit(x_train,y_train,7)
Xtest7 = getPolynomialDataMatrix(x_test, 7)
ytest7 = Xtest7.dot(w7)
plt.plot(x_test, ytest7, 'r')
'''
w10 = getWeightsForPolynomialFit(x_train,y_train,10)
Xtest10 = getPolynomialDataMatrix(x_test, 10)
ytest10 = Xtest10.dot(w10)
plt.plot(x_test, ytest10, 'c')

w12 = getWeightsForPolynomialFit(x_train,y_train,12)
Xtest12 = getPolynomialDataMatrix(x_test, 12)
ytest12 = Xtest12.dot(w12)
plt.plot(x_test, ytest12, 'm')
'''
plt.ylim((-200, 200))
plt.legend(('training points', 'ground truth', '$x^{7}$', '$x^{10}$', '$x^{12}$'), loc = 'lower right')

plt.savefig('polynomial1.png')
plt.show()


'''
E V A L U A T I N G
THE 
M O D E L S
'''
SSEtrain = np.zeros((11,1))
SSEtest = np.zeros((11,1))

# Feel free to use the functions getWeightsForPolynomialFit and getPolynomialDataMatrix
for i in range(1,12):
    
    Xtrain = getPolynomialDataMatrix(x_train, i) 
    Xtest = getPolynomialDataMatrix(x_test, i)
    
    w = getWeightsForPolynomialFit(x_train, y_train, i)  
    
    SSEtrain[i - 1] = np.mean((Xtrain.dot(w) - y_train)**2)
    SSEtest[i - 1] = np.mean((Xtest.dot(w) - y_test)**2)


plt.figure()
plt.semilogy(range(1,12), SSEtrain)
plt.semilogy(range(1,12), SSEtest)
plt.legend(('SSE on training set', 'SSE on test set'))
plt.savefig('polynomial_evaluation.png')
plt.show()