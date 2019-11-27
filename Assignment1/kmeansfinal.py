import numpy as np 
import pandas as pd 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt 
import random 
from copy import deepcopy


def compute_euclidean_distance(vec_1,vec_2, k):
    #return np.linalg.norm(vec_1-vec_2)
    '''
    tempDist=np.sum((X-centroids[:,k])**2,axis=1)
    EuclidianDistance=np.c_[EuclidianDistance,tempDist]
    C=np.argmin(EuclidianDistance,axis=1)+1
    '''
    #Vec 1 will be the dataset and Vec 2 will be the centroid co-ords
    
    vecshape = vec_2.shape[1]
    #print('vecshape');print(vecshape-1)

    #print('k:')
    #print(k)
    Dist=np.sum((vec_1-vec_2[:,k])**2,axis=1)
    return Dist


def initialise_centroids(dataset, k):
    values=dataset.shape[0] 

    centroids_init=np.array([]).reshape(2,0)

    for i in range(k): #Makes as many centroids as are passed into the array 
        rand=random.randint(0,values-1)
        centroids_init=np.c_[centroids_init,dataset[rand]] 
    return centroids_init

def kmeans(dataset, k):
    iterationloops = [] #iterations
    activationResults = np.array([]).reshape(k,0) #Activation function SSE for each 

    X = dataset.iloc[:, [0, 2]].values #Fetches the columns currently being worked with from the imported dataset
    #print('X');print(X)
    m=X.shape[0] #values - in this case 300

    n_iter=15 #todo possibly replace a fixed value for iterations with a while loop that
                    #stops when the change in convergence is lower than a certain amount
    K=k
    centroids=np.array([]).reshape(2,0)

    centroids = initialise_centroids(X, k)

   
    EuclideanDistance=np.array([]).reshape(m,0) #Empty array of size 300,0 where each element is one list of 
                                                #euclidean distances

    for k in range(k):
        '''
        tempDist=np.sum((X-centroids[:,k])**2,axis=1)
        EuclideanDistance=np.c_[EuclidianDistance,tempDist]
        
        EuclideanDistance = compute_euclidean_distance(X, centroids, k)
        '''
        initdistance = compute_euclidean_distance(X, centroids, k)
        #print('Result of InitDistance');print(initdistance)
        EuclideanDistance=np.c_[EuclideanDistance,initdistance]
        #print('Result of printing EuclideanDistance');print(EuclideanDistance)



    CentroidList=np.argmin(EuclideanDistance,axis=1)+1 #+1 because of zero indexing. After getting the distance from each point
                                            #to each centroid, fetches the smallest distance. Whichever centroid
                                            #this distance came from is now the one assigned to the point
    '''
    CentroidList is a list of integers from 1...n where n is the amount of data points.
    The integers range from 1..k and represent to which of the initially randomised clusters each of the
    data points belong to. For example if index 3 is '1' then the third co-ordinate in the dataset is closest
    to the first randomised centroid.
    '''

    OutputDict={} #Initialising dictionary that will contain k keys, each containing the co-ordinates of the
                        #data points assigned to the respective cluster represented by the key index
    for k in range(K):
        OutputDict[k+1]=np.array([]).reshape(2,0) #Initalise empty co-ordinates for the cluster data to be placed


    for i in range(m):
        OutputDict[CentroidList[i]]=np.c_[OutputDict[CentroidList[i]],X[i]] 
        #For each of the values in the centroid
        #fetch the actual co-ordinates of each data point associated with each 
        #of the centroids and feed them into a dictionary
        
     
    for k in range(K):
        OutputDict[k+1]=OutputDict[k+1].T #.T is a numpy operation for ndarrays that just transpose it
        #This results in the X and Y data being arranged in two columns next to each other rather than as adjacent
        #rows 

        
    
    for k in range(K):
        centroids[:,k] = np.mean(OutputDict[k+1],axis=0) #k+1 because zero indexing, I've made clusters start at 1
                                                            #Uses the means of all the co-ordinates assigned to each
                                                            #repsective centroid to get the new co-ordinates for that
                                                            #centroid. This process then repeats until convergence.

                                                            #Centroids consists of two lists, one for x- and y-
                                                            #coordinates
                                                             
        
    
    oldcentroidsl = []
    newcentroisdsl = []

    #Above will be plotted against each other in the end 

    SquaredError = []
    SSE = []


    for i in range(15): #For a fixed amount of iterations, in this case we have it as 100 since it'll do
        #iterationloops[i] = i
     #step 2.a
        tempi = i
        
        EuclideanDistance=np.array([]).reshape(m,0)
        for k in range(K):
            initdistance = compute_euclidean_distance(X, centroids,k)
            EuclideanDistance=np.c_[EuclideanDistance,initdistance]
            
            #print(EuclideanDistance)

        CentroidList=np.argmin(EuclideanDistance,axis=1)+1
        oldcentroids = centroids

        print('Old Centroids');print(oldcentroids)
        oldcentroidsl.append(oldcentroids)
        print('Old Centroids list contents')
        print(oldcentroidsl)
        #SquaredError = SquaredError - centroids
        #rint('Squared Error:');print(SquaredError)


     #step 2.b
        OutputDict={}
        '''
        This is literally just repeating the steps from earlier until iterations are completed
        They are the exact same steps but using previously generated centroids instead of the randomised ones

        The randomisation of the first set of centroids means it needs to work slightly different and cant just 
        be included in this loop
        '''
        for k in range(K):
            OutputDict[k+1]=np.array([]).reshape(2,0)

        for i in range(m):
            '''
            #For each of the values in the centroid
            #fetch the actual co-ordinates of each data point associated with each 
            #of the centroids and feed them into a dictionary
            '''
            OutputDict[CentroidList[i]]=np.c_[OutputDict[CentroidList[i]],X[i]]
            #print('Each iteration p')
            #print(np.c_[OutputDict[CentroidList[i]],X[i]])

     
        for k in range(K):
            OutputDict[k+1]=OutputDict[k+1].T
    
        for k in range(K):
            centroids[:,k]=np.mean(OutputDict[k+1],axis=0)
            #print('Mean iterated ');print(i);print('times')
        
        #print('Old Centroids2');print(tempold)
        #print('New Centroids');print(centroids)
        newcentroisdsl.append(centroids)
        #print('Old - New:');print(centroids-oldcentroids)



        Output=OutputDict
        #print('Output:')
        #print(Output)

    #print('Squared Error:');print(SquaredError)        
    #print('Ultimate centroids list');print(centroidultimatelist)
    #print('Old centroids');print(oldcentroidsl)
    #print('New centroids');print(newcentroisdsl)
    color=['red','blue', 'green']
    labels=['Cluster 1','Cluster 2', 'Cluster 3']
    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    plt.scatter(centroids[0,:],centroids[1,:],s=150,c='black',label='Centroids', marker='X')
    plt.xlabel('Height')
    plt.ylabel('Leg Length')
    plt.legend()
    plt.show()


dataset = pd.read_csv('dog_breeds.csv')
height = dataset['height']
tail = dataset['tail length']
leg = dataset['leg length']
nose = dataset['nose circumference']

k = 3
kmeans(dataset, k)

    
