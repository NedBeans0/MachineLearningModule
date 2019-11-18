import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import sklearn.metrics as metrics
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier


# data_train = pd.DataFrame.from_csv('regression_train.csv')
# data_test = pd.DataFrame.from_csv('regression_test.csv')
data_train = pd.read_csv('week6regressiontrain.csv')
data_test = pd.read_csv('week6regressiontest.csv')

x_train = data_train['x'].values
y_train = data_train['y'].values

x_test = data_test['x'].values
y_test = data_test['y'].values

x_train = x_train.reshape(-1, 1)

x_test = x_test.reshape(-1, 1)

print(x_train.shape, y_train.shape)
plt.figure()
plt.clf()
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.legend(('training points', 'ground truth'))
# plt.hold(True)
plt.savefig('trainingdata.png')
plt.show()

regTree = tree.DecisionTreeRegressor(min_samples_leaf=1, max_depth=None)
regTree = regTree.fit(x_train, y_train)
y_predict = regTree.predict(x_test)

regTree1 = tree.DecisionTreeRegressor(min_samples_leaf=1)
regTree1 = regTree1.fit(x_train, y_train)

y_predict1 = regTree1.predict(x_test)

regTree2 = tree.DecisionTreeRegressor(min_samples_leaf=5)
regTree2 = regTree2.fit(x_train, y_train)
y_predict2 = regTree2.predict(x_test)

regTree3 = tree.DecisionTreeRegressor(min_samples_leaf=10)
regTree3 = regTree3.fit(x_train, y_train)
y_predict3 = regTree3.predict(x_test)


plt.figure()
plt.clf()
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.plot(x_test,y_predict1, 'r', linewidth=2.0)
plt.title('min_samples = 1')
plt.savefig('regressiontrees1.png')


plt.figure()
plt.clf()
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.plot(x_test,y_predict2, 'r',linewidth=2.0)
plt.title('min_samples = 5')
plt.savefig('regressiontrees5.png')


plt.figure()
plt.clf()
plt.plot(x_train,y_train, 'bo')
plt.plot(x_test,y_test, 'g')
plt.plot(x_test,y_predict3, 'r',  linewidth=2.0)
plt.title('min_samples = 10')

plt.savefig('regressiontrees10.png')
plt.show()

#mean squared error
mseTrainTree1 = metrics.mean_squared_error(y_train, regTree1.predict(x_train))
mseTrainTree2 = metrics.mean_squared_error(y_train, regTree2.predict(x_train))
mseTrainTree3 = metrics.mean_squared_error(y_train, regTree3.predict(x_train))

print(mseTrainTree1, mseTrainTree2, mseTrainTree3)

mseTestTree1 = metrics.mean_squared_error(y_test, regTree1.predict(x_test))
mseTestTree2 = metrics.mean_squared_error(y_test, regTree2.predict(x_test))
mseTestTree3 = metrics.mean_squared_error(y_test, regTree3.predict(x_test))

print(mseTestTree1, mseTestTree2, mseTestTree3)

iris = datasets.load_iris()

X = iris.data[:,:2]
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure()
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.savefig('classification_data.png')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
X_train.shape, X_test.shape

decTree = tree.DecisionTreeClassifier(min_samples_leaf=2, max_depth=None)
decTree = decTree.fit(X_train, Y_train)
y_predict = decTree.predict(X_test)
n_classes = 3
plot_colors = "bry"
plot_step = 0.02

clf = DecisionTreeClassifier(min_samples_leaf=5)

clf = clf.fit(X_train, Y_train)  

# create a grid for the two input dimensions
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

plt.figure()
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.axis("tight")
plt.savefig('classification_tree.png')
plt.show()

train_accuracy = metrics.accuracy_score(Y_train, clf.predict(X_train))
test_accuracy = metrics.accuracy_score(Y_test, clf.predict(X_test))

train_accuracy, test_accuracy


'''
E X E R C I S E S
''' 

minSamples = [1,2,3,5,7,10, 15, 20, 50]
train_accuracy = np.zeros((len(minSamples),1))
test_accuracy = np.zeros((len(minSamples),1))


for i in range(0,len(minSamples)):
   
    clf =DecisionTreeClassifier(min_samples_leaf=10)
      
    clf.fit(X_train, Y_train)

    train_accuracy[i] = metrics.accuracy_score(Y_train, clf.predict(X_train))
    test_accuracy[i] = metrics.accuracy_score(Y_test, clf.predict(X_test))
    

plt.figure()
plt.plot(minSamples, train_accuracy, 'b')
plt.plot(minSamples, test_accuracy, 'g')
plt.xlabel('min_samples_per_leaf')
plt.ylabel('accuracy')
plt.legend(('training set', 'test set'))
plt.savefig('classification_minSamples.png')
plt.show()
decForest = ensemble.RandomForestClassifier(n_estimators=10, min_samples_leaf=2, max_depth=None)

decForest = decForest.fit(X_train, Y_train)
y_predict = decForest.predict(X_test)

n_classes = 3
plot_colors = "bry"
plot_step = 0.02

clf = ensemble.RandomForestClassifier(n_estimators=50,min_samples_leaf=10)# add your code here to run the random forests classifier with num_estimators=50 and min_samples_leaf=10

clf = decForest.fit(X_train, Y_train)
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

plt.figure()

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, cmap=plt.cm.Paired)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.axis("tight")
plt.savefig('classification_forest.png')
plt.show()

minSamples = [1,3,5,7,10, 15, 20, 30]

numTrials = 10
train_accuracy_single = np.zeros((len(minSamples),numTrials))
test_accuracy_single = np.zeros((len(minSamples),numTrials))

train_accuracy_mean = np.zeros((len(minSamples),1))
test_accuracy_mean = np.zeros((len(minSamples),1))

train_accuracy_std = np.zeros((len(minSamples),1))
test_accuracy_std = np.zeros((len(minSamples),1))


for i in range(0,len(minSamples)):
    clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=minSamples[i])

    for j in range(0, numTrials):
        clf.fit(X_train, Y_train)

        train_accuracy_single[i,j] = metrics.accuracy_score(Y_train, clf.predict(X_train))
        test_accuracy_single[i, j] = metrics.accuracy_score(Y_test, clf.predict(X_test))

    train_accuracy_mean[i] = np.mean(train_accuracy_single[i,:])
    train_accuracy_std[i] = np.std(train_accuracy_single[i,:])
    
    test_accuracy_mean[i] = np.mean(test_accuracy_single[i,:])
    test_accuracy_std[i] = np.std(test_accuracy_single[i,:])
    
        
plt.figure()
plt.errorbar(minSamples, train_accuracy_mean, yerr = train_accuracy_std)
plt.errorbar(minSamples, test_accuracy_mean, yerr = test_accuracy_std)
plt.savefig('classification_forest_minsamples.png')
plt.show()


numTrees  = [1,5,10,20,40,60,100]

numTrials = 10
train_accuracy_single = np.zeros((len(numTrees),numTrials))
test_accuracy_single = np.zeros((len(numTrees),numTrials))

train_accuracy_mean = np.zeros((len(numTrees),1))
test_accuracy_mean = np.zeros((len(numTrees),1))

train_accuracy_std = np.zeros((len(numTrees),1))
test_accuracy_std = np.zeros((len(numTrees),1))


# put your code here to calculate the average accuracy on the training and 
# on the test set for the given number of trees (n_estimators). 

# the average accuracy on the training set should be saved in train_accuracy_mean 
# and tht on the test set in test_accuracy_mean

    
        
plt.figure()
plt.plot(numTrees, train_accuracy_mean, 'b')
plt.plot(numTrees, test_accuracy_mean, 'g')

plt.savefig('classification_forest_numtrees.png')
plt.show()