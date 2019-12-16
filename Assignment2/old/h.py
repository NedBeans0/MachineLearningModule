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

x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x))
plt.show()