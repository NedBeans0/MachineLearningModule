import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import linalg
from collections import OrderedDict

dataset = pd.read_csv('nucleardata.csv')
pd.options.display.max_columns = 50

#Remove Whitespace from the Headers (this will affect all the Pressure Sensor columns)
dataset.columns = dataset.columns.str.replace(' ', '')


print(dataset.describe()) #For each columns, gives the 

print('SIZE OF DATASET (OBSERVATIONS, FEATURES)');print(dataset.shape)


df = pd.DataFrame(dataset, columns=['Status','Vibration_sensor_1'])
df2 = pd.DataFrame(dataset, columns=['Status','Vibration_sensor_2'])


#Boxplot
df.boxplot(by='Status')
plt.ylabel('Vibration Sensor 1 Values')
plt.show() 
plt.close()


#Density Plot
df2.groupby('Status').plot(kind='kde', ax=plt.gca())
plt.xlabel('Vibration Sensor 2 Values')
plt.title('Density Plot for Vibration Sensor 2')
plt.legend(('Abnormal', 'Normal'), loc = 'upper right')

plt.show()
plt.close() 















