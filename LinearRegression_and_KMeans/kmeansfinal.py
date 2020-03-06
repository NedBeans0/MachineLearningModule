import numpy as np 
import pandas as pd 
from matplotlib import cm 
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt 
import random 
from copy import deepcopy


def compute_euclidean_distance(vec_1,vec_2):
    #Vec_1 = All the data points
    #Vec_2 = The list of centroid coordinates
    print('vec 2 length');print(len(vec_2)) #gets k
    k = (len(vec_2))+1
    valuelength = len(vec_1)
    
    EuclideanDistance=np.array([]).reshape(valuelength,0) #We want a row of distances for each
                                                          #data point - each column is the distance
                                                          #to each centroid in vec_2
    for k in range(k): #For each centroid
        Dist=np.sqrt(np.sum((vec_1-vec_2[:,k])**2,axis=1))   #Get euclidean distance from each point to the centroid
        EuclideanDistance=np.c_[EuclideanDistance,Dist]      #Add a column to the distances array where each row 
                                                             #is the distance from each co-ord to respective centroid

    return EuclideanDistance #Return an array with k columns and a m rows where m is the amount of points


def initialise_centroids(dataset, k):
    values=dataset.shape[0] #Amount of points (300)
    centroids_init=np.array([]).reshape(2,0) #2 rows. First is for centroid x coords. Second is centroid y coords

    for i in range(k): #Makes as many centroids as are passed into the array 
        randcoord =random.randint(0,values-1) #Setup for fetching a random index of coordinates
        centroids_init=np.c_[centroids_init,dataset[randcoord]] #Fill in a column in the array with
                                                                #co-ordinates from a random index of
                                                                #the data being explored
    return centroids_init

def kmeans(dataset, k):
    X = dataset.iloc[:, [0, 1]].values #Fetches the columns currently being worked with from the imported dataset
    m=X.shape[0] #values - in this case 300

    centroids=np.array([]).reshape(2,0)

    centroids = initialise_centroids(X, k)

   
    EuclideanDistance=np.array([]).reshape(m,0) #Empty array of size 300,0 where each element is one list of 
                                                #euclidean distances
   
    initdistance = compute_euclidean_distance(X, centroids)
    EuclideanDistance=np.c_[EuclideanDistance,initdistance]

    n_iter=15 #todo possibly replace a fixed value for iterations with a while loop that
                    #stops when the change in convergence is lower than a certain amount
    K=k


    CentroidList=np.argmin(EuclideanDistance,axis=1)+1 #+1 because of zero indexing. After getting the distance from each point
                                            #to each centroid, fetches the smallest distance. Whichever centroid
                                            #this distance came from is now the one assigned to the point
    print('Centroid List:');print(CentroidList)
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
        #For each of the values in the centroid list
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
                                                             
        


    RSSList = []
    for i in range(8): 
        
        tempi = i
        
        EuclideanDistance=np.array([]).reshape(m,0)
        #for k in range(K):
        initdistance = compute_euclidean_distance(X, centroids)
        #EuclideanDistance=np.c_[EuclideanDistance,initdistance]
        EuclideanDistance = initdistance
        RSS1 = EuclideanDistance

        CentroidList=np.argmin(EuclideanDistance,axis=1)+1
        oldcentroids = centroids
        

        OutputDict={}
        '''
        This is literally just repeating the steps from earlier until iterations are completed
        They are the exact same steps but using previously generated centroids instead of the randomised ones

        The randomisation of the first set of centroids means it needs to work slightly different and cant just 
        be included in this loop
        '''
        for k in range(K):
            OutputDict[k+1]=np.array([]).reshape(2,0)
        #print('Old Centroids 2');print(oldcentroids)

        for i in range(m):
            '''
            #For each of the values in the centroid
            #fetch the actual co-ordinates of each data point associated with each 
            #of the centroids and feed them into a dictionary
            '''
            OutputDict[CentroidList[i]]=np.c_[OutputDict[CentroidList[i]],X[i]]


        for k in range(K):
            OutputDict[k+1]=OutputDict[k+1].T

        #This is where the centroids change
        for k in range(K):
            centroids[:,k]=np.mean(OutputDict[k+1],axis=0)
        
        RSS2 = compute_euclidean_distance(X, centroids)
        RSSTru = RSS1-RSS2
        RSSList.append(RSSTru)

        Output=OutputDict

    #print('RSSList');print(RSSList)
   
    #RSS1 = RSSList[0]
    print('diff1')






    color=['red','blue', 'green']
    labels=['Cluster 1','Cluster 2', 'Cluster 3']
    for k in range(K):
        plt.scatter(Output[k+1][:,0],Output[k+1][:,1],c=color[k],label=labels[k])
    plt.scatter(centroids[0,:],centroids[1,:],s=150,c='black',label='Centroids', marker='X')
    plt.xlabel('Height')
    plt.ylabel('Tail Length')
    plt.legend()
    plt.show()


dataset = pd.read_csv('dog_breeds.csv')
height = dataset['height']
tail = dataset['tail length']
leg = dataset['leg length']
nose = dataset['nose circumference']

k = 3
kmeans(dataset, k)

