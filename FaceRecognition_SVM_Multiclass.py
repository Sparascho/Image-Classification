
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np
import time
import warnings

warnings.simplefilter("ignore")

# Loading the FITW dataset.
faces = fetch_lfw_people(min_faces_per_person = 100)
X = faces.data
y = faces.target

for z in range(0,len(np.unique(y))):
    k=0
    for i in y:
        if i == z:
            k+=1
    print(faces.target_names[z],':',k,'photos')

# Spliting the data into train(60%) and test(40%).
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# SVM parameters.
LinearParameters = {'mdl__max_iter': [10000], 'mdl__C': [1, 10, 100]}
RbfParameters = {'mdl__kernel': ['rbf'], 'mdl__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001], 'mdl__C': [1, 10, 100]}
PolynomialParameters = {'mdl__kernel': ['poly'], 'mdl__degree': [2, 3, 4],'mdl__gamma': [100, 10, 1, 0.1, 0.01, 0.001, 0.0001], 'mdl__C': [1, 10, 50, 100]}

# KNN parameters.
KNNParameters = {'mdl__weights': ['uniform','distance'], 'mdl__metric': ['euclidean','manhattan'], 'mdl__n_neighbors': [1,10,50,100]}

# NCentroid parameters.
NCentroidParameters = {'mdl__metric': ['euclidean','manhattan']}

testCases = [['Linear kernel SVM', LinearSVC(), LinearParameters],
              ['RBF kernel SVM', SVC(), RbfParameters],
              ['Polynomial kernel SVM', SVC(), PolynomialParameters],
              ['K-NearestNeighbors', KNeighborsClassifier(), KNNParameters],
              ['NearestCentroid', NearestCentroid(), NCentroidParameters]]

for method, model, parameters in testCases:
    
    # Constuct a pipeline.
    pipeline = Pipeline([('sampling', SMOTE()),
                     ('scaler', MinMaxScaler()),
                     ('pca', PCA(0.9)),
                     ('mdl', model)])
    print()
    print('----------------------------------------------------------------------------')
    print('Tuning parameters to find the best accuracy using the %s.' % method)
    
    # Execute gridsearch.
    clf = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=10, cv=5, scoring='accuracy')
    clf.fit(x_train, y_train)

    print()
    print("Grid scores on train set:")
    print()
    
    means = clf.cv_results_['mean_test_score']
    params = clf.cv_results_['params']
    fit_times = clf.cv_results_['mean_fit_time']
    a = list(zip(means, params, fit_times))
    k = sorted(a, key = lambda x: x[0])
    k = list(k)[:3]+list(k)[-3:]
    
    for mean, param, fit_time in k:
        print('Score: %0.3f for %r trained in %0.3f sec' % (mean, param, fit_time))
    
    print()    
    print('The best parameters found on train set with a mean score of %0.3f are:' % clf.best_score_)
    print()
    print(clf.best_params_)
    print()
    print('The model is trained on the full train set with the best parameters\nand the scores are computed on the test set.')
    print()
    print('Time to fit the model on the entire train set: %0.3f sec' % clf.refit_time_)
    
    # Calculating the score for the train set.
    y_predicted_train = clf.predict(x_train)
    
    print()
    print('--- Train set scores ---')
    print('Accuracy: %f' % metrics.accuracy_score(y_train, y_predicted_train))
    print('Recall: %f' % metrics.recall_score(y_train, y_predicted_train, average='macro'))
    print('Precision: %f' % metrics.precision_score(y_train, y_predicted_train, average='macro'))
    print('F1: %f' % metrics.f1_score(y_train, y_predicted_train, average='macro'))
    print()
      
    # Calculating the score for the test set.
    start = time.time()
    y_predicted = clf.predict(x_test)
    end = time.time() - start
    
    print('Time to test the model on the test set: %0.3f sec' % end)
    print()   
    print('--- Test set scores ---')  
    print('Accuracy: %f' % metrics.accuracy_score(y_test, y_predicted))
    print('Recall: %f' % metrics.recall_score(y_test, y_predicted, average='macro'))
    print('Precision: %f' % metrics.precision_score(y_test, y_predicted, average='macro'))
    print('F1: %f' % metrics.f1_score(y_test, y_predicted, average='macro'))
    print()
    
    # Classification results per class.
    print(metrics.classification_report(y_test, y_predicted, target_names=faces.target_names))
    