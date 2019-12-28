import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
#from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import linalg
from collections import OrderedDict

dataset = pd.read_csv('nucleardata.csv')

#X = All the data without the Status Column. Independent variables.
#Y = The status column. Dependent variable to predict. Changed to categorical data 0 = Normal, 1 = Abnormal
x = pd.DataFrame(dataset)
x = x.drop('Status', 1)

y = pd.DataFrame(dataset, columns=['Status'])
'''
categoricalConv = {'Normal':0,'Abnormal':1}
y = y.replace(categoricalConv)
'''
#Split randomly into 90% Training Data and 10% Testing Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, train_size=0.9) #test_size = 0.1 equivalent to '=10% of the data' 


#Normalising
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

epochs = [10, 50, 100, 250, 500, 750, 1000, 2000, 5000, 10000]
epochsize = len(epochs)
accuracylist = [] 
for i in range(epochsize):
    #Training a Multi Layer Perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', max_iter=epochs[i], random_state=1)
    print('Current Epochs: ');print(epochs[i])
    setattr(mlp, "out_activation_", "logistic")

    mlp.fit(x_train, y_train)
    mlp_predictions = mlp.predict(x_test)
    print('ANN Classifications:');print(mlp_predictions) 
    iteraccuracy = accuracy_score(y_test, mlp_predictions)
    accuracylist.append(iteraccuracy)

print('ACCURACY LIST:');print(accuracylist)
plt.plot(epochs, accuracylist)
axes = plt.gca()
axes.set_ylim([0,1])


plt.title("ANN: Epochs and Accuracies")
plt.xlabel("Epochs")
plt.ylabel("Accuracies")

plt.show()








'''
#Training a Multi Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', max_iter=1000)
setattr(mlp, "out_activation_", "logistic")

mlp.fit(x_train, y_train)
mlp_predictions = mlp.predict(x_test)
print('ANN Classifications:');print(mlp_predictions) 
iteraccuracy = accuracy_score(y_test, mlp_predictions))
'''
