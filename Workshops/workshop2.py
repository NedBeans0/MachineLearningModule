import numpy as np
import numpy.linalg as linalg
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


'''
                            INTRO / TUTORIALS
'''
'''
#Creating Vectors
a = np.array([1,0,2])
b = np.array([3,2,1])
a.shape
#Adding vectors and computing inner products with the dot function
c = a + b
d = a.dot(c)
d

#Creating matrices
A = np.array([[2, 1, 3], [1, 1 ,2]]) #3x2
B = np.array([[2, 1], [1, 2], [5 ,2]]) #2x3

print(A.shape, B.shape)

#Stacking Vectors as rows or columns in Matrices
X = np.column_stack((a,b))
Y = np.row_stack((a,b))

X,Y

#Add, transpose and multiplying matrices
C = A + B.transpose()
D = C.dot(A.transpose()) # matrix product C * A
C,D

#Multiplying Matrices with Vectors
e = A.dot(a) # this corresponds to A * a
f = a.dot(B) # this corresponds to a^T * B
e, f

#Inverse of a Matrix
AA = A.dot(A.transpose()) # A * A^T ... we can only invert quadratic matrices
AAinv = linalg.inv(AA)

AA, AAinv

#Multiplying with the Inverse
AA.dot(AAinv), AAinv.dot(AA) 

# compute A^-1*b in a more stable way using linalg.solve.
b = np.array([1, 2])
out1 = linalg.solve(AA, b)

out1
'''


'''                          EXERCISES           
'''

A = np.array([[1, 0, 1], [2, 3, 1]])
C = np.array([[1, 0], [2, 3], [1, 5]])
a = np.array([1,2,1])
b = np.array([2,2])

#sol1
Aa = A.dot(a) 
AaM = Aa - b 
AaT = AaM.transpose() 
sol1= AaT.dot(AaM)
sol1 

#sol2
Cb = C * b
#print(Cb)
CbT = Cb.transpose() 
sol2 = CbT.dot(C)
#print(sol2) 

#sol3
C1 = C.transpose().dot(C) 
CT = C.transpose() 
Ctc1 = linalg.solve(C1,CT)
sol3 = Ctc1.dot(a) 
#print(sol3)


'''     
                            LINEAR REGRESSION
''' 



data_train = pd.DataFrame.from_csv('/Users/admin/Downloads/regression_train.csv')
data_test = pd.DataFrame.from_csv('/Users/admin/Downloads/regression_test.csv')

data_train

#Getting the training data as numpy arrays
x_train = data_train['x'].as_matrix()
y_train = data_train['y'].as_matrix()

x_test = data_test['x'].as_matrix()
y_test = data_test['y'].as_matrix()

plt.clf()
plt.plot(x_train,y_train, 'bo')
plt.savefig('trainingdata.png')

#Construct X matrix
Xtilde = np.column_stack((np.ones(x_train.shape), x_train))
Xtilde

#Prediction with hand-picked Betas, in this case 6 and 7
beta_0 = 7
beta_1 = -20
betatilde = np.array([beta_0, beta_1])


Xtest_tilde = np.column_stack((np.ones(x_test.shape), x_test))
ytest_hat = Xtest_tilde.dot(betatilde)
plt.figure()
plt.plot(x_test,ytest_hat, 'r')
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.legend(('predictions', 'training points', 'ground truth'), loc = 'lower right')

plt.savefig('regression_randomPrediction.png')

#The graph we've created has a rather large error, meaning that although the line does make sense to some extent
#(it goes through two of the central points) it is still not as good as we'd like it to be. 
#We can do better. 
#To assess the quality of our prediction, we compute the error as the difference to the training labels.

yhat = Xtilde.dot(betatilde)
error = y_train - yhat
print('Error:')
print(error) 

#We can then compute an error using the summed square error:
SSE = error.dot(error) # The scalar product is also implemented with the dot function (no need for transpose)
print('SSE:')
print(SSE)

def SSE(beta, x, y):
    
    Xtilde = np.column_stack((np.ones(x.shape), x))
    yhat = Xtilde.dot(beta)
    error = y - yhat
    SSE = error.dot(error) 
    return SSE

# specify data points for beta0 and beta1 (from - 200 to 200, using 50 uniformly distributed points)
beta0Array = np.linspace(-200, 200, 50)
beta1Array = np.linspace(-200, 200, 50)

SSEarray = np.zeros((50,50))

for i in range(0,50):
    for j in range(0,50):
        beta = np.array([beta0Array[i], beta1Array[j]])
        SSEarray[i,j] =  SSE(beta, x_train, y_train)

print('SSEArray:')       
print(SSEarray)
#The cell ouputs the SSE for every grid position between -200 and 200 for both dimensions. 
#For a better visualization, we can create a 3D plot. Run the following cell for doing so. 

fig = plt.figure()
ax = fig.gca(projection='3d')

beta0Grid, beta1Grid = np.meshgrid(beta0Array, beta1Array)

# Plot the surface.
surf = ax.plot_surface(beta0Grid, beta1Grid, SSEarray, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('beta0')
plt.ylabel('beta1')
plt.savefig('errorfunction.png')
plt.show()

minIndex = np.argmin(SSEarray)
index1, index2 = np.unravel_index(minIndex, (50,50))

beta0 = beta0Array[index1]
beta1 = beta1Array[index2]

print(beta0)
print(beta1) 
betatilde = np.array([beta_0, beta_1])

Xtest_tilde = np.column_stack((np.ones(x_test.shape), x_test))
ytest_hat = Xtest_tilde.dot(betatilde)


plt.figure()
plt.plot(x_test,ytest_hat, 'r')
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.legend(('predictions', 'training points', 'ground truth'), loc = 'lower right')

plt.savefig('regression_randomPrediction.png')

