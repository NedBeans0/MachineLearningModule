import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random 
from copy import deepcopy

def euclidean(a,b):
    return np.linalg.norm(a-b)

def initialise_centroids(dataset, k):
    X = dataset.iloc[:, [0, 1]].values
    print('X spec');print(X)
    m=X.shape[0] #values
    n=X.shape[1] #features

    n_iter=100
    K=3
    centroids=np.array([]).reshape(2,0)
    
    for i in range(k):
        rand=random.randint(0,m-1)
        centroids=np.c_[centroids,X[rand]]
    Output={}
    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(k):
        tempDist=np.sum((X-centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist]
    C=np.argmin(EuclidianDistance,axis=1)+1

    Y={}
    for k in range(K):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
    for k in range(K):
        Y[k+1]=Y[k+1].T
    
    for k in range(K):
        centroids[:,k]=np.mean(Y[k+1],axis=0)

    for i in range(n_iter):
     #step 2.a
        EuclidianDistance=np.array([]).reshape(m,0)
        for k in range(K):
            tempDist=np.sum((X-centroids[:,k])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b
        Y={}
        for k in range(K):
            Y[k+1]=np.array([]).reshape(2,0)
        for i in range(m):
            Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
        for k in range(K):
            Y[k+1]=Y[k+1].T
    
        for k in range(K):
            centroids[:,k]=np.mean(Y[k+1],axis=0)
        Output=Y
    plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
    plt.xlabel('Tail')
    plt.ylabel('Height')
    plt.legend()
    plt.title('Plot of data points')
    plt.show()
    color=['red','blue', 'green']
    labels=['cluster1','cluster2', 'cluster3']
    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    plt.scatter(centroids[0,:],centroids[1,:],s=100,c='yellow',label='Centroids')
    plt.xlabel('Tail')
    plt.ylabel('Height')
    plt.legend()
    plt.show()


dataset = pd.read_csv('dog_breeds.csv')
height = dataset['height']
tail = dataset['tail length']
leg = dataset['leg length']
nose = dataset['nose circumference']

k = 3
initialise_centroids(dataset, k)

    
