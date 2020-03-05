
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import time

# Loading the Mnist dataset.
mnist = fetch_openml('mnist_784')

X = mnist.data
y = mnist.target

# Changing the data type from string to int.
y = y.astype(np.int8)
    
# Spliting the data into train(60%) and test(40%).
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Making a copy of y_test with the original labels.
y_dec = y_test.copy()

# Converting the multiclass dataset into binary.
for i in range(0, len(y_train)):
    y_train[i] %= 2
for i in range(0, len(y_test)):
    y_test[i] %= 2

# Scaling the data.
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# Appling PCA to maintain 90% of data information.
pca = PCA(.90)
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

# SVM parameters.
LinearParameters = {'max_iter': [100000], 'C': [1, 10, 100 , 1000]}
RbfParameters = {'kernel': ['rbf'], 'gamma': [10, 1, 0.1, 0.01, 0.001], 'C': [1, 10, 100]}
PolynomialParameters = {'kernel': ['poly'], 'degree': [2, 3, 4],'gamma': [10, 1, 0.1, 0.01], 'C': [1, 10, 100]}


# KNN parameters.
KNNParameters = {'weights': ['uniform','distance'], 'metric': ['euclidean','manhattan'], 'n_neighbors': [1,10,100,1000]}

# NCentroid parameters.
NCentroidParameters = {'metric': ['euclidean','manhattan']}

testCases = [['Linear kernel SVM', LinearSVC(), LinearParameters],
              ['RBF kernel SVM', SVC(), RbfParameters],
              ['Polynomial kernel SVM', SVC(), PolynomialParameters],
              ['K-NearestNeighbors', KNeighborsClassifier(), KNNParameters],
              ['NearestCentroid', NearestCentroid(), NCentroidParameters]]


for method, model, parameters in testCases:
    
    print()
    print('----------------------------------------------------------------------------')
    print('Tuning parameters to find the best accuracy using the %s.' % method)
    
    clf = GridSearchCV(model, parameters, n_jobs=-1, verbose=20, cv=5, scoring='accuracy')
    clf.fit(x_train, y_train)
    
    print()
    print("Grid scores on train set:")
    print()
    
    means = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    fit_times = clf.cv_results_['mean_fit_time']
    for mean, param, fit_time in zip(means, params, fit_times):
        print('Score: %0.3f for %r trained in %.2f min.' % (mean, param, int(fit_time/60+(fit_time%60)/100)))
    
    print()    
    print('The best parameters found on train set with a mean score of %0.3f are:' % clf.best_score_)
    print()
    print(clf.best_params_)
    print()
    print('The model is trained on the full train set with the best parameters\nand the scores are computed on the test set.')
    print()
    print('Time to fit the model on the entire train set: %.2f min' % (int(clf.refit_time_/60)+(clf.refit_time_%60)/100))
    
    # Calculating  the score metrics for the train set.
    y_predicted_train = clf.predict(x_train)
    
    print()
    print('--- Train set scores ---')
    print('Accuracy: %f' % metrics.accuracy_score(y_train, y_predicted_train))
    print('Recall: %f' % metrics.recall_score(y_train, y_predicted_train, average='macro'))
    print('Precision: %f' % metrics.precision_score(y_train, y_predicted_train, average='macro'))
    print('F1: %f' % metrics.f1_score(y_train, y_predicted_train, average='macro'))
    print()
      
    # Calculating  the score metrics for the test set.
    start = time.time()
    y_predicted = clf.predict(x_test)
    end = time.time() - start
    
    print('Time to test the model on the test set: %.2f min' % (int(end/60)+(end%60)/100))
    print()   
    print('--- Test set scores ---')  
    print('Accuracy: %f' % metrics.accuracy_score(y_test, y_predicted))
    print('Recall: %f' % metrics.recall_score(y_test, y_predicted, average='macro'))
    print('Precision: %f' % metrics.precision_score(y_test, y_predicted, average='macro'))
    print('F1: %f' % metrics.f1_score(y_test, y_predicted, average='macro'))
    print()
    
    # Inverse PCA.
    x_testPCA = pca.inverse_transform(x_test)
    
    # Plotting the correct predictions
    fig, ax = plt.subplots(2, 10, figsize=(12,3))
    p = 0
    for i in range(len(y_test)):
        if p>9:
           break 
        elif y_predicted[i] == y_test[i] and y_dec[i] == p:
            ax[0,p].imshow(x_testPCA[i].reshape(28, 28), cmap=matplotlib.cm.binary, interpolation='nearest')
            ax[0,p].set(xticks=[], yticks=[])
            ax[0,p].set_xlabel(y_predicted[i], size=15)
            p+=1 
    
    # Plotting the incorrect predictions
    p = 0
    for i in range(len(y_test)):
        if p>9:
           break 
        elif y_predicted[i] != y_test[i] and y_dec[i] == p:
            ax[1,p].imshow(x_testPCA[i].reshape(28, 28), cmap=matplotlib.cm.binary, interpolation='nearest')
            ax[1,p].set(xticks=[], yticks=[])
            ax[1,p].set_xlabel(y_predicted[i], color='red', size=15)
            p+=1  
    fig.suptitle('Predictions (Incorrect labels in red)', size=20)
   
    plt.show()
