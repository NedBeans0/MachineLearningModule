import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random 

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
    '''
    datasetnp = np.asarray(dataset)

    y_min= datasetnp[:,0].min()
    y_max =datasetnp[:,0].max()
    
    x_min= datasetnp[0,:].min()
    x_max =datasetnp[0,:].max()

    centroidList = []
    
    centroidList.append((random.uniform(x_min,x_max),(random.uniform(y_min,y_max))))
    centroidList.append((random.uniform(x_min,x_max),(random.uniform(y_min,y_max))))
    centroidList.append((random.uniform(x_min,x_max),(random.uniform(y_min,y_max))))


    #centroidList.append(random.uniform(x_min,x_max))
    #centroidList.append(random.uniform(x_min,x_max))
    '''
    centroidslist = np.zeros((k,2)) #4 dimensions
    for i in range(k):
        centroidslist[(i,0)] = np.random.choice(height)
        centroidslist[(i,1)] = np.random.choice(tail)
    

    return centroidslist

def kmeans(dataset, centroids, k):
    centroids, cluster_assigned = 'h'
    return centroids, cluster_assigned

dataset = pd.read_csv('dog_breeds.csv',header=None).values
#Height, Tail Length, Leg Length, Nose Circumference


height = dataset[1:,0] 
TailLength = dataset[1:,1]
LegLength = dataset[1:,2]
NoseCircumference = dataset[1:,3]

#Task 1 - Scatter Plot of X:Height and Y: Tail Length

centroid_dataset1 = np.zeros((300,2))
centroid_dataset1[0:,0] = height 
centroid_dataset1[:,1] = TailLength

heightplot = centroid_dataset1[:,0]
tailplot = centroid_dataset1[:,1]
#print(centroid_dataset1)

k=2

centroids1 = initialise_centroids(centroid_dataset1, k)


centroids1t = initialise_centroids(centroid_dataset1, k)
centroid1 = list(centroids1t[0]) #Convert to list from tuples
centroid2 = list(centroids1t[1])
#centroid3 = list(centroids1t[2])


distances1 = []
for i in range(0,300):
    distance = compute_euclidean_distance(centroid1, centroid_dataset1[i])
    distances1.append(distance)

distances2 = []
for i in range(0,300):
    distance = compute_euclidean_distance(centroid2, centroid_dataset1[i])
    distances2.append(distance)
'''
distances3 = []
for i in range(0,300):
    distance = compute_euclidean_distance(centroid3[0], centroid_dataset1[i])
    distances3.append(distance)
'''
cluster1points = []
cluster2points = []

for i in range(0,300):
    if distances1[i]<distances2[i]:
        cluster1points.append(centroid_dataset1[i])

for i in range(0,300):
    if distances1[i]>distances2[i]:
        cluster2points.append(centroid_dataset1[i])

print('Cluster 1 points amount');print(len(cluster1points))
print('Cluster 2 points amount');print(len(cluster2points))

cluster1meantotals = 0 
for i in range(len(cluster1points)):
    cluster1meantotals += cluster1points[i]
cluster1mean = cluster1meantotals/len(cluster1points)
print('Current Cluster 1 Mean:');print(cluster1mean)

cluster2meantotals = 0
for i in range(len(cluster2points)):
    cluster2meantotals += cluster2points[i]
cluster2mean = cluster2meantotals/len(cluster2points)
print('Current Cluster 2 Mean:');print(cluster2mean)

plt.plot(tailplot, heightplot, 'bo')
plt.plot(centroid1, centroid2, 'ro')

plt.show()





'''
distance1 = compute_euclidean_distance(centroids1[0], centroid_dataset1[0])
print('Centroid 1');print(centroid1)
print('Coordinate1');print(pointlist)
print('Distance:');print(distance1)
'''
