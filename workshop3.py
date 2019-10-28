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
#plt.hold(True)
plt.savefig('trainingdata.png')
#plt.show()

#Preparing the Matrices
X = np.column_stack((np.ones(x_train.shape), x_train))
print(X)

A = np.array([[1, 0.5], [0.5, 1]])
a = np.array([[1], [0]])

# specify data points for x0 and x1 (from - 5 to 5, using 51 uniformly distributed points)
x0Array = np.linspace(-5, 5, 51)
x1Array = np.linspace(-5, 5, 51)


Earray = np.zeros((51,51))

for i in range(0,50):
    for j in range(0,50):
        Earray[i,j] = 




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

fig = plt.figure()
ax = fig.gca(projection='3d')

x00Grid, x1Grid = np.meshgrid(x0Array, x1Array)

# Plot the surface.
surf = ax.plot_surface(x0Grid, x1Grid, Earray, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('beta0')
plt.ylabel('beta1')
plt.savefig('errorfunction.png')
plt.show()
