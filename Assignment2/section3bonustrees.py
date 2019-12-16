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

#Split randomly into 90% Training Data and 10% Testing Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, train_size=0.9) #test_size = 0.1 equivalent to '=10% of the data' 


#Normalising
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

trees = [10, 50, 100, 250, 500, 750, 1000, 2000, 5000]
treessize = len(trees)
accuracylist = [] 
accuracylist2 = []

for i in range(treessize):
    #Training a Random Forest Classifier 
    rfc = RandomForestClassifier(n_estimators=trees[i], min_samples_leaf=5)
    rfc.fit(x_train, y_train)
    rfc_predictions = rfc.predict(x_test)
    print('Tree Classifications:');print(rfc_predictions) 
    iteraccuracy = accuracy_score(y_test, rfc_predictions)
    accuracylist.append(iteraccuracy)

print('ACCURACY LIST:');print(accuracylist)
plt.plot(trees, accuracylist)

for i in range(treessize):
    #Training a Random Forest Classifier 
    rfc = RandomForestClassifier(n_estimators=trees[i], min_samples_leaf=50)
    rfc.fit(x_train, y_train)
    rfc_predictions = rfc.predict(x_test)
    print('Tree Classifications:');print(rfc_predictions) 
    iteraccuracy = accuracy_score(y_test, rfc_predictions)
    accuracylist2.append(iteraccuracy)
plt.plot(trees, accuracylist2)


axes = plt.gca()
axes.set_ylim([0,1])

plt.title("RFC: Tree Counts and Accuracies")
plt.xlabel("Tree Counts")
plt.ylabel("Accuracies")
plt.legend(('5 Min Split', '50 Min Split'), loc = 'lower right')

plt.show()