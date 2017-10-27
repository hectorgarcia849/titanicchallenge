# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 22:35:07 2017

@author: hecto
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from graph.curves import plot_learning_curve, plot_validation_curve, plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def printf(format, *args):
    sys.stdout.write(format % args)

def percentage_breakdown(num, den):
    pb = round(num/float(den), 4)
    return pb
    
if __name__ == '__main__':#this is necessary for multithreading...
    
	#Welcome
    print ("...............................................................................")
    print ("Welcome To Titantic ML Challenge")
    print ("...............................................................................\n")
	
	print("This is a command line program that verbosely trains machine learning algorithms.")
	print("It provides some successful results in predicting the survival rate based on the passenger manifest.")
	print("Note that program may appear to hang, but it is probably not.  Ensure to hit enter and close graphs to continue the program!")

	input("Press Enter to continue...\n")

    #exploring the data
    print ("...............................................................................")
    print ("Exploring and Cleaning the Data")
    print ("...............................................................................\n")
    
    print ("<Loading the Titantic Challenge training and test .csv files>\n")
    df_train = pd.read_csv("C:\\Users\\hecto\\Documents\\Programming\\Kaggle\\Titantic Challenge\\train.csv")
    df_submit_test = pd.read_csv("C:\\Users\\hecto\\Documents\\Programming\\Kaggle\\Titantic Challenge\\test.csv")
    
    print ("Identified the Survived Feature as the Target label in the data set.....")
    print ("<Splitting the Y Target label from the data set and storing in seperate array y>\n")
    print ("<X stores the training data, X_test_submit stores the test data>\n")
    
    y = df_train["Survived"];
    X = df_train
    X_test_submit = df_submit_test
    del X["Survived"];
    
    input("Press Enter to continue...\n")
    
    print ("Remove Unneeded features from the dataset...\n")
    print ("<Removing feature PassengerID>")
    print ("PassengerID is an arbitrarily assigned number to each passenger\n")
    print ("<Removing feature Ticket Number>")
    print ("Ticket Number Ticket data does not seem to provide any information \nthat we can't get from the Passenger Class and Fare features.\n")
    print ("<Removing Feature Cabin>")
    print ("Cabin is missing data too much data.  In cabin (687 of 892 rows are \nmissing this data for this feature), besides the location of the cabin \na passenger was staying at does not equate to the location of the passenger \nwhen the Titanic was sinking.\n")
    
    del X["PassengerId"], X["Ticket"], X["Cabin"], X["Name"]
    del X_test_submit["PassengerId"], X_test_submit["Ticket"], X_test_submit["Cabin"], X_test_submit["Name"]
    
    input("Press Enter to continue...\n")
    
    print ("Search data for missing values: NaNs (not a number), blanks, and nulls...\n")
    
    print ("Training data results for null values: \n")
    print (X.isnull().sum(), "\n")
    print ("Test data results for missing values: \n")
    print (X_test_submit.isnull().sum(), "\n")
    
    mean_age_sib2 = round(X.groupby(X['SibSp'] <= 2.0).median()['Age'][1], 2)
    mean_age_sib3 = round(X.groupby(X['SibSp'] > 2.0).median()['Age'][1], 2)
    
    print ("Interpolation, imputing mean, median and mode...\n")
    print ("For Age, imputing the median age is appropriate since the value measures...")
    print ("However, grouping age by Sibsp to get median age.")
    print ("If person has Sipsp <= 2, imputed median age: ", mean_age_sib2)
    print ("If person has Sipsp > 2, imputed median age: ", mean_age_sib3)
    print ("<Using this median to impute into missing values for the Age feature in training and test set>\n")
    
    X['Age'] = X.groupby(X['SibSp'] <= 2.0)['Age'].fillna(mean_age_sib2)
    X['Age'] = X.groupby(X['SibSp'] > 2.0)['Age'].fillna(mean_age_sib3)
    X = X.round({'Age': 1})
    
    X_test_submit['Age'] = X_test_submit.groupby(X['SibSp'] <= 2.0)['Age'].fillna(mean_age_sib2)
    X_test_submit['Age'] = X_test_submit.groupby(X['SibSp'] > 2.0)['Age'].fillna(mean_age_sib3)
    
    input("Press Enter to continue...\n")
    
    print ("For Fare, mean imputation is appropriate since the value measures...")
    print ("Mean Fare by passenger class in training set(rounded to two decimal places): ")
    
    class_1 = X[(X["Pclass"] == 1)]
    class_2 = X[(X["Pclass"] == 2)]
    class_3 = X[(X["Pclass"] == 3)]
    fare1 = round(class_1['Fare'].mean(), 2) 
    fare2 = round(class_2['Fare'].mean(), 2)         
    fare3 = round(class_3['Fare'].mean(), 2)  
                      
    print ("Class 1: ", fare1 , "Class 2: ", fare2, "Class 3: ", fare3)
    print ("<Using this mean to impute into missing values for the Fare feature in training and test set>\n")
    
    a = X[(X['Fare'].isnull())]
    b = X_test_submit[(X_test_submit['Fare'].isnull())]
    
    print ("Records with missing fare value:\n")
    print ("From training set: \n", a)
    print ("From test set: \n", b)
          
    X_test_submit['Fare'] = X_test_submit['Fare'].fillna(fare3)
         
    #if(X[(X['Fare'].isnull())] and X['Pclass'] == 1):
    #    X['Fare'].fillna(fare1)
    #elif(X[(X['Fare'].isnull())] and X['Pclass'] == 2):
    #    X['Fare'].fillna(fare2)
    #else:
    #    X['Fare'].fillna(fare3)
    # 
    #if(X_test_submit[(X_test_submit['Fare'].isnull())] and X_test_submit['Pclass'] == 1):
    #    X_test_submit['Fare'].fillna(fare1)
    #elif(X_test_submit[(X_test_submit['Fare'].isnull())] and X_test_submit['Pclass'] == 2):
    #    X_test_submit['Fare'].fillna(fare2)
    #else:
    #    X_test_submit['Fare'].fillna(fare3)     
    
    print ("Passenger of class 3, using mean value for class 3 fare")
    
    input("Press Enter to continue...\n")
    
    print ("For Embarked, mode imputation is appropriate since it is a categorical feature...")
    print ("Mode of Embarked in training set: ", X['Embarked'].value_counts().index[0])
    print ("<Using this meode to impute into missing values for the Embarked feature in training and test set>/n")
    
    mode = X["Embarked"].value_counts().index[0] #index 0, because that is max occurence.  Auto-sorted.
    X['Embarked'] = X['Embarked'].fillna(mode)
    
    input("Press Enter to continue...\n")
    #no more missing values verified:
    print ("Verifying that all missing values cleaned up...")
    print ("Training data results for null values: \n")
    print (X.isnull().sum())
    print ("Test data results for null values: \n")
    print (X_test_submit.isnull().sum())
    
    print ("Visualizng and Describing the data\n")
    print ("<Converting Categorial features: PClass, Sex, Embarked,and (target) Survived into dtype=category for descriptive statistics>") 
    
    
    for col in ['Pclass','Sex', 'Embarked']:
        temp1 = pd.get_dummies(X[col]) 
        temp2 = pd.get_dummies(X_test_submit[col]) 
        del X[col], X_test_submit[col]
        X = pd.concat([X, temp1], axis=1)
        X_test_submit = pd.concat([X_test_submit, temp2], axis=1)
    
    X.rename(columns={1:'Pclass_1', 2:'Pclass_2', 3:'Pclass_3', 'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southamption'}, inplace=True)
    X_test_submit.rename(columns={1:'Pclass_1', 2: 'Pclass_2', 3:'Pclass_3', 'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southamption'}, inplace=True)
    
    print ("Convert integers to floating numbers, this is necessary for scilean Random Forest Algorithm")
    
    X[["SibSp","Parch"]] = X[["SibSp","Parch"]].astype('float64')    
    X_test_submit[["SibSp","Parch"]] = X_test_submit[["SibSp","Parch"]].astype('float64') 
    
    print (X.head(10))
    print (X_test_submit.head(10))
     
    descriptive_stats = X.describe()
    
    print (descriptive_stats.round(4))
    
    print ("Encode categorical variables PClass, Sex, Embarked...\n")
    
    
    print ("Data Transformation, last step to clean data - feature scaling applied to the data set")
    X_std = X
    X_test_submit_std = X_test_submit
    
    stdsc = StandardScaler(copy=True, with_mean=True, with_std=True) #must use standard scaler based on params from training, these  same params are appplied to the test data after.
    X_std[['Age','Fare']] = stdsc.fit_transform(X[['Age','Fare']]) #fit and transform, to get params from X
    X_test_submit_std[['Age','Fare']] = stdsc.transform(X_test_submit[['Age','Fare']])#only transform, because we will use the params from X
    
    input("Press Enter to continue...\n")
    
    print ("...............................................................................")
    print ("Split Data into Training and Test set")
    print ("...............................................................................\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    
    print ("Breakdown of training data, test data:")
    print ("Samples in training data: ", X_train.shape[0])
    print ("Samples in testing data: ", X_test.shape[0])
    print ("80% in training data, 20% in test data")
    
    input("Press Enter to continue...\n")

    print ("...............................................................................")
    print ("Feature Scaling")
    print ("...............................................................................\n")
    
    X_train_std = X_train.copy()
    X_test_std = X_test.copy()
    X_test_submit_std = X_test_submit.copy()
    
    stdsc = StandardScaler(copy=True, with_mean=True, with_std=True) #must use standard scaler based on params from training, these  same params are appplied to the test data after.
    X_train_std[['Age','Fare']] = stdsc.fit_transform(X_train_std[['Age','Fare']]) #fit and transform, to get params from X
    X_test_std[['Age','Fare']]   = stdsc.fit_transform(X_test_std[['Age','Fare']])
    X_test_submit_std[['Age','Fare']] = stdsc.transform(X_test_submit_std[['Age','Fare']])#only transform, because we will use the params from X

    input("Press Enter to continue...\n")

    print ("...............................................................................")
    print ("Principal Component Analysis")
    print ("...............................................................................\n")
    
    #model tuning optimizing hyperparameters (validation curves)
    
    #Optimizing Principal Components
    pipe_lr = Pipeline([('pca', PCA()),
                        ('clf', LogisticRegression(penalty='l2', 
                                                   random_state=1,
                                                   verbose=1))])
    
    comp_range = range(1,12,1)
    train_scores, test_scores = validation_curve(estimator=pipe_lr, 
                                                 X=X_train_std, 
                                                 y=y_train,
                                                 param_name='pca__n_components',
                                                 param_range=comp_range,
                                                 cv=10)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    title = "Validation Curve for Logistic Regression"
    param_title = "Principal Components Analysis"
    
    plot_validation_curve(title, 
                          param_title, 
                          comp_range, 
                          train_mean, 
                          train_std, 
                          test_mean, 
                          test_std)
    
     
    idx = test_mean.argmax() #an array of the cumulative variance explained ratio, find first max
    opt_components = comp_range[idx]

    print ("The number of components that the improves computational efficiency\n")
    print ("with limited information loss: ", opt_components)

    
    input("Press Enter to continue...\n")

    print ("...............................................................................")
    print ("The Machine Learning Pipelines")
    print ("...............................................................................\n")
    
    pipe_lr = Pipeline([('pca', PCA(n_components = opt_components)),
                        ('clf', LogisticRegression(penalty='l2', 
                                                   random_state=1,
                                                   verbose=1))])
    
    
    
    print ("Model tuning Regularization for Logistic Regression\n\n")
    
    input("Press Enter to continue...\n")

    
    #Optimizing the regularization term: C = 1/lambda
    c_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(estimator=pipe_lr, 
                                                 X=X_train_std, 
                                                 y=y_train,
                                                 param_name='clf__C',
                                                 param_range=c_range,
                                                 cv=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    title = "Validation Curve for Logistic Regression"
    param_title = "Parameter C"
    
    plot_validation_curve(title, param_title, c_range, train_mean, train_std, test_mean, test_std)
    
    idx = test_mean.argmax()
    opt_C = c_range[idx]

    print ("From the Validation curve we can see that a Parameter C above 10 is optimal./n")
    print ("The Parameter C that produces the highest level of model acuracy: ", opt_C)
    
    input("Press Enter to continue...\n")
    
    print ("Running logistic regression with new hyperparameter values...")
    
    pipe_lr = Pipeline([('pca', PCA(n_components = opt_components)),
                        ('clf', LogisticRegression(penalty='l2', 
                                                   random_state=1,
                                                   C = opt_C))])
        
    pipe_lr.fit(X_train_std, y_train)
    
    print("Logistic Regression with Hyperparameters Tuned")
    printf('Test Accuracy: %.3f\n' % pipe_lr.score(X_test_std, y_test)) #now that cv done, can use test data
    
    
    #using the stratified k-fold method for cv, using sci-kit learn
    scores = cross_val_score(estimator=pipe_lr,
                             X=X_train_std,
                             y=y_train,
                             cv=10,
                             n_jobs=-1)
            
    print('CV accuracy scores: %s' % scores)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
            
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, 
                                                            X=X_train_std, 
                                                            y=y_train, 
                                                            train_sizes=np.linspace(0.1, 1.0, 10), #we set train_sizes=np.linspace(0.1, 1.0, 10) to use 10 evenly spaced relative intervals for the training set sizes.
                                                            cv=10, #k=10, set via the cv param
                                                            n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    title = "Learning Curve for Logistic Regression"
    
    plot_learning_curve(title, train_sizes, train_mean, train_std, test_mean, test_std)
    
    input("Press Enter to continue...\n")
    
    print ("Model tuning hyperparameters Gamma and C for SVM\n\n")
    
    pipe_svc = Pipeline([('pca', PCA(n_components = opt_components)),
                         ('clf', SVC(random_state=1, verbose=1))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    #set the param_grid parameter of GridSearchCV to a list of dictionaries to 
    #specify the parameters that we'd want to tune.
    #For the linear SVM, we only evaluated the inverse regularization parameter C;
    # for the RBF kernel SVM, we tuned both the C and gamma parameter. 
    
    #Note that the gamma parameter is specific to kernel SVMs.
    #Intuitively, the gamma parameter defines how far the influence of a single training 
    #example reaches, with low values meaning ‘far’ and high values meaning ‘close’. 
    #The gamma parameters can be seen as the inverse of the radius of influence of 
    #samples selected by the model as support vectors.
    
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, 
                  {'clf__C': param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
    
    #initialize gridsearch object to train and tune a support vector machine (SVM) pipeline.
    #Grid Search will run all combinations of for C and gamma using k-fold cross-validation
             
    gs = GridSearchCV(estimator=pipe_svc, 
                      param_grid=param_grid, 
                      scoring='accuracy', 
                      cv=10,
                      verbose=1,
                      n_jobs=-1) #n_jobs=-1 uses all cores
    gs = gs.fit(X_train_std, y_train)
    print("Results for SVM")
    print("Best Score: ", gs.best_score_)
    print("Best Params: ", gs.best_params_)
    
    #Finally, we will use the independent test dataset to estimate the performance 
    #of the best selected model, which is available via the best_estimator_ attribute 
    #of the GridSearchCV object
        
    clf = gs.best_estimator_
    best_parameters = clf.get_params()
    clf.fit(X_train_std, y_train)
    
    print('Test accuracy: %.3f' % clf.score(X_test_std, y_test))
    
    y_pred = clf.predict(X_test_std)    
    print (classification_report(y_test, y_pred))
    
    #print ('Best parameters set:', best_parameters)
    
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    
    plot_confusion_matrix(confmat)
    
    
    #Algorithm selection with nested Cross-validation
    #this technique uses an inner loop to find the optimal hyperparameters
    #for a given learning algorithm in the outer loop.  Very useful.
    
    #first an svm
#    gs = GridSearchCV(estimator=pipe_svc,
#                      param_grid=param_grid,
#                      scoring='accuracy',
#                      cv=10,
#                      n_jobs=4)
#    
#    scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
#    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
#    
#    plot_learning_curve(title, train_sizes, train_mean, train_std, test_mean, test_std)
#    
    
    #input("Press Enter to continue...\n")
    
    print ("Decision Tree Learning \n\n")
    
    #then try a decision tree classifier
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                      param_grid=[{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}],
                                  scoring='accuracy',
                                  cv=10,
                                  n_jobs=-1)
    scores = cross_val_score(gs,
                             X_train,
                             y_train,
                             scoring='accuracy',
                             cv=10)
    
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    pipe_svc.fit(X_train, y_train)
    y_pred = pipe_svc.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    
    plot_confusion_matrix(confmat)
    
    #precision, recall and f1-score, measuring the relevance of the model
    
    print (classification_report(y_test, y_pred))


    print ("Random Forest")

    forest = RandomForestClassifier(criterion='entropy',
                                    n_estimators=10,
                                    random_state=1, 
                                    n_jobs=-1) #jobs indicates how many cores to use.
    
    gs = GridSearchCV(estimator = forest, 
                      param_grid = [{'max_depth': [1,2,3,4,5,6,7,None]}], 
                                    scoring ='accuracy', 
                                    cv=10)
    
    scores = cross_val_score(gs,
                             X_train,
                             y_train,
                             scoring='accuracy',
                             cv=10)
    print("Results for Decision Tree")
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    forest.fit(X_train, y_train)
    accuarcy_score = forest.score(X_test, y_test)
    print('Accuracy: %.2f\n' % accuarcy_score)
    
    y_pred = forest.predict(X_test)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    
    plot_confusion_matrix(confmat)
    
    y_pred = forest.predict(X_test_submit)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.to_csv("submission.csv")
    
    
    print ("Neural Network")
    
    
    clf = MLPClassifier(solver='lbgfs', 
                        #alpha=1e-5, #alpha is regularization term for L2
                        #hidden_layer_sizes=(5, 2), #hiddenlayer1 = 5 neurons, hiddenlayer2 = 2 neurons 
                        random_state=1) 
    
    param_grid = [{'clf__alpha': 10.0 ** -np.arange(1,7), 'clf__hidden_layer_sizes': [(3,3),(4,3),(5,3)]}] 
    
    gs = GridSearchCV(estimator = clf, 
                      param_grid = param_grid,
                      scoring ='accuracy', 
                      cv=10)
    scores = cross_val_score(gs,
                             X_train_std,
                             y_train,
                             scoring='accuracy',
                             cv=10,
                             n_jobs=-1)
    print ("Results for Neural Network:")
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    clf.fit(X_train_std, y_train)
    y_pred = clf.predict(X_test_std)
    print (classification_report(y_test, y_pred))