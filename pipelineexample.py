# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 15:22:05 2017

@author: hecto
"""
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

def printf(format, *args):
    sys.stdout.write(format % args)

if __name__ == '__main__':#this is necessary for multithreading...
    
    
    #in df breast tumour data. Column 0: id number, column 1: malignant or benign, 
    #column 2-31: real-value features computed from digitized images 
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    
    #store the 30 features in an array x, encode string malignant, benign
    X = df.loc[:, 2:].values
    y = df.loc[:, 1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    #split data 80% training, 20% test
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    
    #Instead of going through the fitting and transformation steps for the training and test 
    #dataset separately, we can chain the StandardScaler, PCA, and LogisticRegression objects 
    #in a pipeline:
        
    pipe_lr = Pipeline([('scl',
                         StandardScaler()), 
                        ('pca', PCA(n_components=2)),
                        ('clf', LogisticRegression(random_state=1))])
    
    pipe_lr.fit(X_train, y_train)
    printf('Test Accuracy: %.3f\n' % pipe_lr.score(X_test, y_test))
    
    #The Pipeline object takes a list of tuples as input, where the first value 
    #in each tuple is an arbitrary identifier string that we can use to access 
    #the individual elements in the pipeline, and the second element in every 
    #tuple is a scikit-learn transformer or estimator.
    
    #The intermediate steps in a pipeline constitute scikit-learn transformers, and 
    #the last step is an estimator
    
    #When we executed the fit method on the pipeline pipe_lr, the StandardScaler 
    #performed fit and transform on the training data, and the transformed training 
    #data was then passed onto the next object in the pipeline, the PCA. Similar to 
    #the previous step, PCA also executed fit and transform on the scaled input data 
    #and passed it to the final element of the pipeline, the estimator.
    
    #Using scikit to fold training data for cross validation.  The method below
    #is the stratified K-fold cross validatio method.  This is used for model selection
    #and finding the hyperparameters.
    
    
    #using the stratified k-fold method for cross validation, an illustrative more manual approach
    #kfold = StratifiedKFold(y=y_train, n_folds=10, random_state=1)
    #scores = []
    
    #for k, (train, test) in enumerate(kfold):
        #pipe_lr.fit(X_train[train], y_train[train])
        #score = pipe_lr.score(X_train[test], y_train[test])
        #scores.append(score)
        #print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y_train[train]), score))
    
    #using the stratified k-fold method for cv, using sci-kit learn
    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             n_jobs=4)
        
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    #learning curve plotting
    
    pipe_lr = Pipeline([('scl', StandardScaler()), 
                        ('clf', LogisticRegression(penalty='l2', random_state=0))])
    
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, 
                                                            X=X_train, 
                                                            y=y_train, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), #we set train_sizes=np.linspace(0.1, 1.0, 10) to use 10 evenly spaced relative intervals for the training set sizes.
                                                            cv=10, #k=10, set via the cv param
                                                            n_jobs=4)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(train_sizes, 
             train_mean, 
             color='blue', 
             marker='o',
             markersize=5,
             label='training accuracy')
    
    #we add the standard deviation of the average accuracies to the plot using 
    #the fill_between function to indicate the variance of the estimate
    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, 
                     color='blue')
    plt.plot(train_sizes, 
             test_mean,
             color='green', 
             linestyle='--',
             marker='s', 
             markersize=5,
             label='validation accuracy')
    plt.fill_between(train_sizes, 
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, 
                     color='green')
    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()
    
    #Validation curve used to improve performace of a model by addressing overfitting/underfitting
    
    #Inside the validation_curve function, we specified the parameter that we wanted to 
    #evaluate. In this case, it is C, the inverse regularization parameter of the 
    #LogisticRegression classifier, which we wrote as 'clf__C' to access the LogisticRegression
    #object inside the scikit-learn pipeline for a specified value range that we set via the 
    #param_range parameter
    
    param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(estimator=pipe_lr, 
                                                 X=X_train, 
                                                 y=y_train,
                                                 param_name='clf__C',
                                                 param_range=param_range,
                                                 cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.plot(param_range, 
             train_mean, 
             color='blue', 
             marker='o',
             markersize=5,
             label='training accuracy')
    plt.fill_between(param_range, 
                     train_mean + train_std, 
                     train_mean - train_std, 
                     alpha=0.15,
                     color='blue')
    plt.plot(param_range, 
             test_mean, 
             color='green', 
             linestyle='--',
             marker='s', 
             markersize=5,
             label='validation accuracy')
    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, 
                     color='green')
    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.show()
    
    #tuning hyperparameters via grid-search.  Specify a list of values for the various 
    #hyperparameters and the computer evaluates the performance of the different 
    #combinations to obtain an optimal set.
    
    pipe_svc = Pipeline([('scl', StandardScaler()),('clf', SVC(random_state=1))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    #set the param_grid parameter of GridSearchCV to a list of dictionaries to 
    #specify the parameters that we'd want to tune.
    #For the linear SVM, we only evaluated the inverse regularization parameter C;
    # for the RBF kernel SVM, we tuned both the C and gamma parameter. 
    #Note that the gamma parameter is specific to kernel SVMs.
    
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, 
                  {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
    
    #initialize gridsearch object to train and tune a support vector machine (SVM) pipeline.              
    gs = GridSearchCV(estimator=pipe_svc, 
                      param_grid=param_grid, 
                      scoring='accuracy', 
                      cv=10, 
                      n_jobs=4) #n_jobs=-1 uses all cores
    gs = gs.fit(X_train, y_train)
    
    print(gs.best_score_)
    print(gs.best_params_)
    
    #Finally, we will use the independent test dataset to estimate the performance 
    #of the best selected model, which is available via the best_estimator_ attribute 
    #of the GridSearchCV object
    
    clf = gs.best_estimator_
    clf.fit(X_train, y_train)
    print('Test accuracy: %.3f' % clf.score(X_test, y_test))
    
    #Algorithm selection with nested Cross-validation
    #this technique uses an inner loop to find the optimal hyperparameters
    #for a given learning algorithm in the outer loop.  Very useful.
    
    #first an svm
    gs = GridSearchCV(estimator=pipe_svc,
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=4)
    scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    #then try a decision tree classifier
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                      param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                                  scoring='accuracy',
                                  cv=5)
    scores = cross_val_score(gs,
                             X_train,
                             y_train,
                             scoring='accuracy',
                             cv=5)
    
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    
    #plot a diagram of confusion matrix
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=j, 
                    y=i,
                    s=confmat[i, j],
                    va='center', 
                    ha='center')
            
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.show()
    
    #precision, recall and f1-score, measuring the relevance of the model
    
    
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    print('F1: %.3f' % f1_score(y_true=y_test, y_pred=y_pred))