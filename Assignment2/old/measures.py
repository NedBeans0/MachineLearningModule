import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import linalg
from collections import OrderedDict

def meanList(data):
    means = np.mean(data)
    return means

        
def stdList(data):
    stdevs = np.std(data) 
    return stdevs


def minList(data):
    mins = np.min(data)
    return mins


def maxList(data):
    maxs = np.max(data)
    return maxs



dataset = pd.read_csv('nucleardata.csv')

powersensors = []
powersens1 = dataset['Power_range_sensor_1'];powersensors.append(powersens1)
powersens2 = dataset['Power_range_sensor_2'];powersensors.append(powersens2)
powersens3 = dataset['Power_range_sensor_3'];powersensors.append(powersens3) 
powersens4 = dataset['Power_range_sensor_4'];powersensors.append(powersens4)

pressuresensors = []
pressuresens1 = dataset['Pressure _sensor_1'];pressuresensors.append(pressuresens1)
pressuresens2 = dataset['Pressure _sensor_2'];pressuresensors.append(pressuresens2)
pressuresens3 = dataset['Pressure _sensor_3'];pressuresensors.append(pressuresens3)
pressuresens4 = dataset['Pressure _sensor_4'];pressuresensors.append(pressuresens4)

vibesensors = []
vibesens1 = dataset['Vibration_sensor_1'];vibesensors.append(vibesens1)
vibesens2 = dataset['Vibration_sensor_2'];vibesensors.append(vibesens2)
vibesens3 = dataset['Vibration_sensor_3'];vibesensors.append(vibesens3)
vibesens4 = dataset['Vibration_sensor_4'];vibesensors.append(vibesens4)


powermeans = []
for sensor in powersensors:
    meanofsens = meanList(sensor)
    powermeans.append(meanofsens)

pressuremeans = [] 
for sensor in pressuresensors:
    meanofsens = meanList(sensor) 
    pressuremeans.append(meanofsens)

vibemeans = [] 
for sensor in vibesensors:
    meanofsens = meanList(sensor) 
    vibemeans.append(meanofsens)



powerstd = []
for sensor in powersensors:
    sensStd = stdList(sensor)
    powerstd.append(sensStd)

pressurestd = [] 
for sensor in pressuresensors:
    sensStd = stdList(sensor) 
    pressurestd.append(sensStd)

vibestd = [] 
for sensor in vibesensors:
    sensStd = stdList(sensor) 
    vibestd.append(sensStd)



powermin = []
for sensor in powersensors:
    sensMin = minList(sensor)
    powermin.append(sensMin)

pressuremin = [] 
for sensor in pressuresensors:
    sensMin = minList(sensor) 
    pressuremin.append(sensMin)

vibemin = [] 
for sensor in vibesensors:
    sensMin = minList(sensor) 
    vibemin.append(sensMin)



powermax = []
for sensor in powersensors:
    sensMax = maxList(sensor)
    powermax.append(sensMax)

pressuremax = [] 
for sensor in pressuresensors:
    sensMax = maxList(sensor) 
    pressuremax.append(sensMax)

vibemax = [] 
for sensor in vibesensors:
    sensMax = maxList(sensor) 
    vibemax.append(sensMax)


'''
print('PowerMeans');print(powermeans)
print('PressMeans');print(pressuremeans)
print('VibeMeans');print(vibemeans)

print('PowerStd');print(powerstd)
print('PressStd');print(pressurestd)
print('VibeStd');print(vibestd)

print('PowerMin');print(powermin)
print('PressMin');print(pressuremin)
print('VibeMin');print(vibemin)

print('PowerMax');print(powermax)
print('PressMax');print(pressuremax)

print('VibeMax');print(vibemax)
'''

summary = dataset.describe()
print(summary)



df = pd.DataFrame(dataset, columns=['Status','Vibration_sensor_1'])
df2 = pd.DataFrame(dataset, columns=['Status','Vibration_sensor_2'])


#Remove Whitespace from the Headers
#if whitespace remove

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









'''

categoricalConv = {'Normal':0,'Abnormal':1}
df = df.replace(categoricalConv)


abs = df.index[df['Status'] == 1].tolist()
firstab = abs[0]

rowtotal = df.shape[0]

normals = df.iloc[0:firstab]
abnormals = df.iloc[firstab:rowtotal]
'''
'''
boxplot = df.boxplot(column=['Status', 'Vibration_sensor_1'])
plt.show()

categoricalConv = {'Normal':0,'Abnormal':1}
df = df.replace(categoricalConv)

abs = df.index[df['Status'] == 1].tolist()
firstab = abs[0]
rowtotal = df.shape[0]


normals = df2.iloc[0:firstab]
abnormals1 = df2.iloc[firstab:rowtotal]
abnormals = abnormals1.rename(columns={'Vibration_sensor_2': 'Vibration_sensor_2a'})

dat1 = normals.loc[:, ['Vibration_sensor_2']]
dat2 = abnormals.loc[:, ['Vibration_sensor_2a']]
densdata = pd.concat([dat1, dat2.reset_index()], axis=1)
densdata = densdata.drop('index', 1)

print('datadens:');print(densdata)
densdata.plot.kde()



plt.show() 
plt.close()


'''





