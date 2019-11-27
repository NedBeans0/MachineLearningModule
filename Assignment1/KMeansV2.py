import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random 
import copy 


def compute_euclidean_distance(vec_1, vec_2):
    '''
    Euclidean distance is: for each value in x and y, sum their squared differences and square root.
    This could be done with a loop quite easily but numpy has functions to make it more readable.

    This function is used to evaluate to which centroid (and therefore to which cluster) a point in the vector belongs.
    It is sorted into the cluster associated with the closest centroid (i.e. the one with the lowest Euclidean
    distance from the value).
    '''
    #distance = np.sqrt(np.sum((vec_1-vec_2)**2))
    distance = np.linalg.norm(vec_1 - vec_2)
        


    return distance


def initialise_centroids(dataset, k):
    '''
    A centroid is a point in the dataset around which the clustering will take form.

    Initially these are randomised, and eventually optimised through an iterative process of calculating 
    the distance from each point to each centroid.

    K is the amount of clusters being created and since each cluster has one centroid only, K here can indicate
    the amount of centroids
    '''

    
    X = dataset.iloc[:, [0, 1]].values
    print('X spec');print(X)
    m=X.shape[0] #values
    n=X.shape[1] #features

    
    centroids=np.array([]).reshape(2,0)
    for i in range(k):
        rand=random.randint(0,m-1)
        centroids=np.c_[centroids,X[rand]]
    return centroids
    


def iterativeImprovement(data, centroid1, centroid2):
    distances1 = []
    for i in range(0,300):
        distance = compute_euclidean_distance(centroid1, data[i])
        distances1.append(distance)

    distances2 = []
    for i in range(0,300):
        distance = compute_euclidean_distance(centroid2, data[i])
        distances2.append(distance)
    
    cluster1points = []
    cluster2points = []

    for i in range(0,300):
        if distances1[i]<distances2[i]:
            cluster1points.append(list(data[i]))

    for i in range(0,300):
        if distances1[i]>distances2[i]:
            cluster2points.append(list(data[i]))

    print('Cluster 1 points amount');print(len(cluster1points))
    print('Cluster 2 points amount');print(len(cluster2points))
    print('Cluster 1 Points');print(cluster1points)
    print('Cluster 2 Points');print(cluster2points)


    
    for k in range(2):
        Y[k+1]=np.array([]).reshape(2,0)
    for i in range(vec_1):
        Y[C[i]]=np.c_[Y[C[i]],X[i]]
    for k in range(K):
        Y[k+1]=Y[k+1].T
    for k in range(K):
        centroids[:,k]=np.mean(Y[k+1],axis=0)
    
    
    cluster1mean = [sum(x)/len(x) for x in zip(*cluster1points)]
    print('Cluster 1 Mean:');print(cluster1mean)
    cluster2mean = [sum(x)/len(x) for x in zip(*cluster2points)]
    print('Cluster 2 Mean:');print(cluster2mean)


    '''
    plt.plot(data[:,0], data[:,1], 'bo')
    plt.plot(centroid2, centroid1, 'ro')
    plt.plot(cluster1mean, cluster2mean, 'yo')
    '''
    #plt.show()
    
    newcentroids=[]
    newcentroids.append(cluster1mean)
    newcentroids.append(cluster2mean)
    return newcentroids

    




    



def kmeans(dataset, k):
    
    centroids = initialise_centroids(dataset, k)
    dataset1 = pd.read_csv('dog_breeds.csv',header=None).values

    print(centroids)
    print(dataset)


    y1 = centroids[0][0]
    x1 = centroids[0][1]

    y2 = centroids[1][0]
    x2 = centroids[1][1]



    centroid1l = []
    centroid1l.append(y1);centroid1l.append(x1)

    centroid2l = []
    centroid2l.append(y2);centroid2l.append(x2)



    print('centroid1');print(centroid1l)
    print('centroid2');print(centroid2l)

    height = dataset1[1:,0] 
    TailLength = dataset1[1:,1]

    workingdata = np.zeros((300,2))
    workingdata[:,0] = TailLength
    workingdata[:,1] = height

    newCentroids = iterativeImprovement(workingdata, centroid1l, centroid2l)



    


    print('It is currently plotting');print(newCentroids[0]);print(newCentroids[1])
    print('And original centroids:');print(centroid2l);print(centroid1l)

    plt.plot(workingdata[:,0], workingdata[:,1], 'bo')
    plt.plot(centroid2l, centroid1l, 'ro')
    plt.show()

    plt.plot(workingdata[:,0], workingdata[:,1], 'bo')
    plt.plot(newCentroids[0], newCentroids[1], 'go')
    plt.show()

    
  


    '''
    for i in range(0,300):
        distance = compute_euclidean_distance(centroid1l, dataset[i])
        distances1.append(distance)
    '''
    



dataset = pd.read_csv('dog_breeds.csv')
height = dataset['height']
tail = dataset['tail length']
leg = dataset['leg length']
nose = dataset['nose circumference']

k = 2
kmeans(dataset, k)
