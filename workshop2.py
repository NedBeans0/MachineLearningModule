import numpy as np
import numpy.linalg as linalg

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


sol1start = ((A*a)-b)
sol1t= sol1start.transpose()

sol1 = sol1t*((A*a)-b)
print(sol1)
