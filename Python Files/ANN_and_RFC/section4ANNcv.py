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
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import cross_val_score
from scipy import linalg
from collections import OrderedDict

def KFoldCV(layersizes, x, y, k):
    mlp = MLPClassifier(hidden_layer_sizes=(layersizes,layersizes), activation='logistic')
    setattr(mlp, "out_activation_", "logistic")
    
    #10 folds. Accuracy as CV score measurement. n_jobs to use all CPU cores.
    mlpcv = cross_val_score(mlp, x, y, cv=k,scoring='accuracy',n_jobs=-1) 
    print('CROSS VAL SCORE FOR ',layersizes, ':');print(mlpcv)
    mplmean = np.mean(mlpcv) #Mean of each of the K accuracies.
    mplstd = np.std(mlpcv)   #Std. Dev of each of the K accuracies.
    print('MEAN OF ANN CV FOR ', layersizes, ':');print(mplmean)
    print('STD.DEV OF ANN CV FOR ', layersizes, ':');print(mplstd, '\n\n')

dataset = pd.read_csv('nucleardata.csv')
dataset = shuffle(dataset)

#X = All the data without the Status Column. Independ ent variables.
#Y = The status column. Dependent variable to predict. Changed to categorical data 0 = Normal, 1 = Abnormal
x = pd.DataFrame(dataset)
x = x.drop('Status', 1)
y = pd.DataFrame(dataset, columns=['Status'])
# Evaluation with K-Fold Cross-Validation (Where K=10)
k=10
KFoldCV(50, x, y, k)
KFoldCV(500, x, y, k)
KFoldCV(1000, x, y, k)

