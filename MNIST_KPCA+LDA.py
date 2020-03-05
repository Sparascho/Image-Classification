from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import metrics
import numpy as np
import time

# Loading the Mnist dataset.
mnist = fetch_openml('mnist_784')

X = mnist.data[:12000]
y = mnist.target[:12000]

# Changing the data type from string to int.
y = y.astype(np.int8)
    
# Spliting the data into train(60%) and test(40%).
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# KPCA parameters.
LinearKPcaParameters = {'kpca__n_components': [87,154,331], 'kpca__kernel': ['linear']}

RbfKPcaParameters = {'kpca__n_components': [100,500,1500], 'kpca__gamma': [0.1,0.01,0.001,0.0001],'kpca__kernel': ['rbf']}

PolyKPcaParameters = {'kpca__n_components': [100,500,1500], 'kpca__gamma': [0.1,0.01,0.001,0.0001],
                  'kpca__degree': [2,3,4], 'kpca__kernel': ['poly']}

# LDA parameters.
LdaParameters = {'lda__n_components': [9]}

# KNN parameters.
KNNParameters = {'mdl__weights': ['uniform','distance'], 'mdl__metric': ['euclidean','manhattan'], 'mdl__n_neighbors': [1,10,50,100]}

# NCentroid parameters.
NCentroidParameters = {'mdl__metric': ['euclidean','manhattan']}

testCases = [['K-NearestNeighbors', KNeighborsClassifier(), KNNParameters],
             ['NearestCentroid', NearestCentroid(), NCentroidParameters]]


for method, model, parameters in testCases:
    
    # Constuct a pipeline.
    pipeline = Pipeline([('scaler', MinMaxScaler()),
                     ('kpca', KernelPCA()),
                     ('lda', LDA()),
                     ('mdl', model)])
    
    # Test every KPCA kernel.
    for kpcaParams in [LinearKPcaParameters,RbfKPcaParameters,PolyKPcaParameters]:
        
        # Prepare the pipeline parameters.
        PipeParams = {}
        PipeParams.update(parameters)
        PipeParams.update(kpcaParams)
        PipeParams.update(LdaParameters)
        
        print()
        print('----------------------------------------------------------------------------')
        print('Tuning parameters to find the best accuracy using the %s.' % method)
        
        # Execute gridsearch.
        clf = GridSearchCV(pipeline, PipeParams, n_jobs=6, verbose=5, cv=3, scoring='accuracy')
        clf.fit(x_train, y_train)
          
        means = clf.cv_results_['mean_test_score']
        params = clf.cv_results_['params']
        fit_times = clf.cv_results_['mean_fit_time']
        a = list(zip(means, params, fit_times))
        k = sorted(a, key = lambda x: x[0])
        k = list(k)[:3]+list(k)[-3:]
        
        for mean, param, fit_time in k:
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
        
        # Classification results per class.
        print(metrics.classification_report(y_test, y_predicted))