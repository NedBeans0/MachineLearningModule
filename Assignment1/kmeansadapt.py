import numpy as np 
import pandas as pd 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt 
import random 
from copy import deepcopy

def euclidean(a,b):
    return np.linalg.norm(a-b)

def randomise_centroids(dataset, k):
    print('h')




def kmeans(dataset, k):
    X = dataset.iloc[:, [0, 1]].values #Fetches the columns currently being worked with from the imported dataset
    print('X spec');print(X)
    m=X.shape[0] #values - in this case 300
    n=X.shape[1] #features - in this case 2 for each graph

    iterations=100 #todo possibly replace a fixed value for iterations with a while loop that
                    #stops when the change in convergence is lower than a certain amount
    
    centroids=np.array([]).reshape(2,0) #empty array for the centroids we're about to randomly generate
    
    for i in range(k):
        rand=random.randint(0,m-1)
        centroids=np.c_[centroids,X[rand]]
    Output={}



    EuclidianDistance=np.array([]).reshape(m,0)
    for k in range(k):
        tempDist=np.sum((X-centroids[:,k])**2,axis=1)
        EuclidianDistance=np.c_[EuclidianDistance,tempDist] #np.c_ arranges as coordinates with (x,y) axes
    C=np.argmin(EuclidianDistance,axis=1)+1


    
    Y={}
    for k in range(k):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(m):
        Y[C[i]]=np.c_[Y[C[i]],X[i]]
     
    for k in range(k):
        Y[k+1]=Y[k+1].T
    
    for k in range(k):
        centroids[:,k]=np.mean(Y[k+1],axis=0)

    for i in range(iterations): #For a fixed amount of iterations, in this case we have it as 100 since it'll do
     #step 2.a
        EuclidianDistance=np.array([]).reshape(m,0)
        for k in range(k):
            tempDist=np.sum((X-centroids[:,k])**2,axis=1)
            EuclidianDistance=np.c_[EuclidianDistance,tempDist]
        C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b
        itersoluton2={}
        for k in range(k):
            itersoluton2[k+1]=np.array([]).reshape(2,0)
        for i in range(m):
            itersoluton2[C[i]]=np.c_[itersoluton2[C[i]],X[i]]
     
        for k in range(k):
            itersoluton2[k+1]=itersoluton2[k+1].T
    
        for k in range(k):
            centroids[:,k]=np.mean(itersoluton2[k+1],axis=0)
        Output=itersoluton2

    plt.scatter(X[:,0],X[:,1],c='black',label='unclustered data')
    plt.xlabel('Tail')
    plt.ylabel('Height')
    plt.legend()
    plt.show()
    color=['red','blue','green']
    labels=['cluster1','cluster2','cluster3']
    for k in range(k):
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
kmeans(dataset, k)
