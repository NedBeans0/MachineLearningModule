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

def KFoldCV(treecount, x, y, k):
    rfc20 = RandomForestClassifier(n_estimators=treecount, min_samples_leaf=5)
    rfc20cv = cross_val_score(rfc20, x, y, cv=k,scoring='accuracy', n_jobs=-1)
    print('CROSS VAL SCORE FOR ',treecount,':');print(rfc20cv)
    rfc20mean = np.mean(rfc20cv)
    rfc20var = np.std(rfc20cv)
    print('MEAN OF RFC CV FOR ',treecount,':');print(rfc20mean)
    print('STD. DEVIATION OF RFC CV FOR ',treecount,':');print(rfc20var,'\n\n')

dataset = pd.read_csv('nucleardata.csv')
dataset = shuffle(dataset)

#X = All the data without the Status Column. Independent variables.
#Y = The status column. Dependent variable to predict. Changed to categorical data 0 = Normal, 1 = Abnormal
x = pd.DataFrame(dataset)
x = x.drop('Status', 1)
y = pd.DataFrame(dataset, columns=['Status'])


# Evaluation with K-Fold Cross-Validation (Where K=10)
k=10 
KFoldCV(20,x,y,k)
KFoldCV(500,x,y,k)
KFoldCV(10000,x,y,k)


