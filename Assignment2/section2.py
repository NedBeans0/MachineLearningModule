import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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


#Training a Multi Layer Perceptron
mlp = MLPClassifier(hidden_layer_sizes=(500, 500), activation='logistic', max_iter=10000)
setattr(mlp, "out_activation_", "logistic")

mlp.fit(x_train, y_train)
mlp_predictions = mlp.predict(x_test)
print('ANN Classifications:');print(mlp_predictions) 

#Evaluating the Multi Layer Perceptron
print("\nConfusion Matrix for ANN: \n", confusion_matrix(y_test, mlp_predictions))
print("Accuracy score for ANN: ", accuracy_score(y_test, mlp_predictions))






#Training a Random Forest Classifier 
rfc = RandomForestClassifier(n_estimators=1000, min_samples_leaf=5)
rfc.fit(x_train, y_train)
rfc_predictions = rfc.predict(x_test)
print('RFC Classifications w/ 5:');print(rfc_predictions) 
#Evaluating the 5 Minimum Node RFC
print("\nConfusion Matrix for RFC1: \n", confusion_matrix(y_test, rfc_predictions))
print("Accuracy score for RFC1: ", accuracy_score(y_test, rfc_predictions))




rfc2 = RandomForestClassifier(n_estimators=1000, min_samples_leaf=50)
rfc2.fit(x_train, y_train)
rfc_predictions2 = rfc2.predict(x_test)
print('RFC Classifications w/ 500:');print(rfc_predictions2) 
#Evaluating the 50 Minimum Node RFC
print("\nConfusion Matrix for RFC2: \n", confusion_matrix(y_test, rfc_predictions2))
print("Accuracy score for RFC2: ", accuracy_score(y_test, rfc_predictions2))













'''
testlength = len(x_test) #Could also be y_test we just want the length of the testing set (100) in this case

#For evaluation purposes let's turn it into a numpy array
mlphits = 0
y_testarr = y_test.to_numpy() 
for i in range(testlength):
    if mlp_predictions[i] == y_testarr[i]:
        mlphits += 1

mlp_error_rate = (mlphits/testlength) 
print('ANN ACCURACY:');print(mlp_error_rate)
'''












