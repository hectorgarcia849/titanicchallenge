# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:25:05 2017

@author: hecto
"""
import sys

# visualization
from graph.curves import plot_learning_curve, plot_validation_curve, plot_confusion_matrix, plot_roc_curve, plot_decision_boundary
import graphviz
import pydotplus
import seaborn as sns
import matplotlib.pyplot as plt

#ui
from menu import menu

#data analysis and wrangling
import collections
import itertools
import numpy as np
import pandas as pd
import re

#preprocessing
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as sm 
from sklearn.preprocessing import PolynomialFeatures

#model selection
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score, GridSearchCV

#machine learning algorithms
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_regression, SelectPercentile, RFECV, SelectKBest

#machine learning reports
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, r2_score, f1_score

#dimensionality reduction
from sklearn.decomposition import PCA


def printf(format, *args):
    sys.stdout.write(format % args)
    
def presentSection(SectionName):
    print ("\n###" + SectionName + "###\n")    
    
def presentStepTo(StepMessage):
    input("\nPress Enter to " + StepMessage +"...\n")

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return "Unknown"
    
def groupByTitle(title):
    if(title == 'Capt' or title == 'Col' or title == 'Major'):
        return "military_title"
    if( title == "Don" or title == 'Jonkheer' or title == 'Sir' or title == "Countess" or title == 'Lady'):
        return "noble_title"
    if(title == "Dr" or title == "Rev"):
        return "professional_title"
    if(title == "Master" or title == "Mr" or title == "Miss" or title == "Mlle" or title == "Mme" or title == "Mrs" or title == "Ms"):
        return "common_title"  
    else:
        return "unknown_Title"
        
def extractAndGroupTitles(name):
    return groupByTitle(get_title(name))
 
        
if __name__ == '__main__':    
        
    ###Data Pre-Processing and Cleaning###    
    
    presentSection("Data Pre-Processing and Cleaning")
    presentStepTo("Import the data set")
        
    df_train = pd.read_csv("C:\\Users\\hecto\\Documents\\Programming\\Kaggle\\Titantic Challenge\\train.csv")
    print(df_train.head())
    
    presentStepTo("Split into data into independent variables and dependent variable")
    X = df_train.copy()
    y = df_train["Survived"].copy()
    del X["Survived"]
    
    print (X.head())
    print (y.head())
    
    presentStepTo("remove features that cannot be used because too much missing data or not relevant information")
    printf ("Removing %s, %s, %s", "PassengerId", "Ticket", "Cabin\n")
    
    del X["PassengerId"], X["Ticket"], X["Cabin"]
    
    presentStepTo("to extract titles from names")
    printf("Though names in themselves not useful, titles can be extracted from the name feature to create new column")
    
    titles = X["Name"].apply(get_title).unique()
    X["Title"] = X["Name"].apply(extractAndGroupTitles)
    print(X["Title"].head())
    
    print("No longer have use for Name feature as Title extracted and is its own feature")
    
    del X["Name"]
    
    presentSection("Imputation")
    
    presentStepTo("Search for Missing Values")
    print (X.isnull().sum(), "\n")
    printf ("y has %d missing values\n\n", y.isnull().sum())
    
    presentStepTo("impute location for Embarked")
    print("Only two observations missing data for Embarked, using mode imputation since very few missing and is categorical")
    embarked_mode = X["Embarked"].value_counts().index[0] #index 0, because that is max occurence.  Auto-sorted.
    printf("The mode of Embarked location is %s\n", embarked_mode)
    X['Embarked'] = X['Embarked'].fillna(embarked_mode)
    
    presentStepTo("encode categorical variables, required for Regression Imputation...")
    
    for col in ['Pclass','Sex', 'Embarked', 'Title']:
        temp1 = pd.get_dummies(X[col]) 
        del X[col]
        X = pd.concat([X, temp1], axis=1)
#    del X['Title'] may not even be worth having title
            
    X.rename(columns={1:'Pclass_1', 2:'Pclass_2', 3:'Pclass_3', 'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southampton'}, inplace=True)
    
    print(X.head())
        
    presentStepTo("train linear regression model to predict and impute age")
    
    #make copy
    X_temp = X.copy()
    
    #recreate classes
    X_temp['Pclass'] = 0
    X_temp['Pclass'] = X_temp.apply(lambda row: 1 if row['Pclass_1'] == 1 else 2 if row['Pclass_2'] == 1 else 3, axis=1).astype(int)
    del X_temp['Pclass_1'], X_temp['Pclass_2'], X_temp['Pclass_3']

    #remove all rows with missing values, split into y and X
    X_temp = X_temp.dropna()
        
    #re-order columns, x0 first column
    cols = X_temp.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    X_temp = X_temp[cols]
    
    #split into y values
    y_temp = X_temp["Age"]
    del X_temp["Age"]
    
    feature_names_quant = ['Pclass', 'SibSp', 'Parch', 'Fare']
    feature_names_categorical = ['female', 'male', 'Cherbourg', 'Queenstown', 'Southampton',
       'common_title', 'noble_title', 'military_title', 'professional_title']
    
    X_temp_quant = X_temp[feature_names_quant]
    
    #First search for polynomial features only includes quantitative data.  
    #In this search, we include both interactions and and powers
    X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=0)
    
    #polynomial regression
    poly_reg = PolynomialFeatures(degree=2)
    X_temp_train_q = poly_reg.fit_transform(X_temp_train[feature_names_quant])
    poly_feature_names_quant = poly_reg.get_feature_names(feature_names_quant)
    
    #Selecting by univariate measures, get top 50 percentile
    #removes any highly correlated variables
    #uses greedy search to choose which variables to keep by f-score and p-value
    
    selector_f = SelectPercentile(f_regression, percentile=50)
    selector_f.fit(X_temp_train_q, y_temp_train)
    
    for n,s,p in zip(poly_feature_names_quant, selector_f.scores_, selector_f.pvalues_):
        print ("F-score: %3.2f, p-value: %3.2f for feature %s " % (s,p,n))
        
    regression = LinearRegression()    
    greedy_selector = RFECV(estimator=regression, cv=10,
                     scoring='neg_mean_squared_error')
    greedy_selector.fit(X_temp_train_q, y_temp_train)    
    print('Optimal number of features: %d' % greedy_selector.n_features_)
    
    #create dataframe with optimal polynomial features
    X_temp_train_df = pd.DataFrame(X_temp_train_q)
    X_temp_train_df.columns = poly_feature_names_quant
    X_temp_train_df = X_temp_train_df.loc[:, greedy_selector.support_]
        
    age_regressor = LinearRegression()
    age_regressor = age_regressor.fit(X_temp_train_df, y_temp_train)
    
    #prep X test data, perform polynomial features selection then only use features as already output by
    #greedy selection
    
    X_temp_test_q = pd.DataFrame(poly_reg.fit_transform(X_temp_test[feature_names_quant]))
    X_temp_test_q.columns = poly_feature_names_quant
    X_temp_test_q = X_temp_test_q.loc[:, greedy_selector.support_]
    
    y_pred = age_regressor.predict(X_temp_test_q)
        
    print(r2_score(y_temp_test, y_pred))
    
    #Second search for polynomial features includes qualitatitive data, i.e. categorical variables
    #Will only search for interaction effects
    
    poly_reg = PolynomialFeatures(degree=2, interaction_only=True)

    #create new table from new polynomial features concat with categorical features
    
    X_temp_train_c = poly_reg.fit_transform(X_temp_train[feature_names_categorical])
    poly_feature_names_categorical = poly_reg.get_feature_names(feature_names_categorical)
    X_temp_train_c = pd.DataFrame(X_temp_train_c)
    X_temp_train_c.columns = poly_feature_names_categorical
    X_temp_train_df = pd.concat([X_temp_train_df, X_temp_train_c], axis=1)
        
    selector_f = SelectPercentile(f_regression, percentile=50)
    selector_f.fit(X_temp_train_df, y_temp_train)
    
    for n,s,p in zip(X_temp_train_df.columns, selector_f.scores_, selector_f.pvalues_):
        print ("F-score: %3.2f, p-value: %3.2f for feature %s " % (s,p,n))

    regression = LinearRegression()    
    greedy_selector = RFECV(estimator=regression, cv=10,
                     scoring='neg_mean_squared_error')
    greedy_selector.fit(X_temp_train_df, y_temp_train)    
    print('Optimal number of features: %d' % greedy_selector.n_features_)
    
    X_temp_train_df =  X_temp_train_df.loc[:, greedy_selector.support_]
    
    X_temp_test_c = pd.DataFrame(poly_reg.fit_transform(X_temp_test[feature_names_categorical]))
    X_temp_test_c.columns = poly_feature_names_categorical
    
    X_temp_test = pd.concat([X_temp_test_q, X_temp_test_c], axis=1)
    X_temp_test = X_temp_test.loc[:, greedy_selector.support_]

    
    age_regressor = LinearRegression()
    age_regressor = age_regressor.fit(X_temp_train_df, y_temp_train)
    
    y_pred = age_regressor.predict(X_temp_test)
        
    print(r2_score(y_temp_test, y_pred))

    X_age_regression_df = X.copy()
    X_age_series = X_age_regression_df['Age']
    X_age_regression_df['Pclass'] = X_age_regression_df.apply(lambda row: 1 if row['Pclass_1'] == 1 else 2 if row['Pclass_2'] == 1 else 3, axis=1).astype(int)
    del X_age_regression_df['Pclass_1'], X_age_regression_df['Pclass_2'], X_age_regression_df['Pclass_3']
    
    poly_reg = PolynomialFeatures(degree=2)
    X_age_regression_q = poly_reg.fit_transform(X_age_regression_df[feature_names_quant])
    X_age_regression_q = pd.DataFrame(X_age_regression_q)
    X_age_regression_q.columns = poly_reg.get_feature_names(feature_names_quant)
    del X_age_regression_q['1']
    
    poly_reg = PolynomialFeatures(degree=2, interaction_only=False)
    X_age_regression_c = poly_reg.fit_transform(X_age_regression_df[feature_names_categorical])
    X_age_regression_c = pd.DataFrame(X_age_regression_c)
    X_age_regression_c.columns = poly_reg.get_feature_names(feature_names_categorical)
    del X_age_regression_c['1']
    
    X_age_regression_df = pd.concat([X_age_regression_q, X_age_regression_c], axis=1).loc[:, greedy_selector.support_]
 
    X_age_regression_df['Age'] = X_age_series
        
    age_pred = age_regressor.predict(X_age_regression_df.loc[X_age_regression_df['Age'].isnull(), X_age_regression_df.columns != 'Age'])
    
    X.loc[X['Age'].isnull(), "Age"] = pd.Series(age_pred, index= X.loc[X['Age'].isnull(), "Age"].index)
    X.loc[X['Age'] < 0, "Age"] = 0                
    
    
    presentStepTo("verify there are no more missing values")
    
    print(X.isnull().sum())
    
    presentSection("Correlations")

    presentStepTo("to reveiw correlations between features, print correlation matrix and ranked table")

    correlations = pd.concat([X, y], axis=1).corr()
    print(correlations)
    
    features = correlations.columns.tolist()
    paired_features = list(itertools.combinations(features, 2))
    corr_df = pd.DataFrame([])
    corr_df['feature_pair'] = []
    corr_df['pearson_correlation'] = []
    for pair in paired_features:
        corr_df.loc[len(corr_df)] = [pair, correlations[pair[0]][pair[1]]]
    
    corr_df = corr_df.sort_values(by="pearson_correlation", ascending=False)
    corr_survived = corr_df[corr_df['feature_pair'].apply(lambda x: x[1] == 'Survived')]
    
    print(corr_df)
    print(corr_survived)
    
    fig, ax = plt.subplots(figsize=(30,30))
    sns.heatmap(correlations, 
        xticklabels=correlations.columns,
        yticklabels=correlations.columns, 
        annot=True,
        ax=ax)
    
    plt.show()
    
    presentStepTo("remove redundancies by targeting variables that present multicollinearity, i.e. dealing with the dummy variable trap")
    
    del X['female'], X['Queenstown'], X['common_title'], X['Pclass_3']
    
    
#    presentStepTo("check for interaction effects")
#
#
#    logistic_reg = LogisticRegression()  
#    interaction = PolynomialFeatures(degree=2, include_bias=False)
#    X_inter = pd.DataFrame(interaction.fit_transform(X), columns=interaction.get_feature_names(X.columns))
#    
#    greedy_selector = RFECV(estimator=logistic_reg, cv=10, scoring='accuracy')
#    greedy_selector.fit(X_inter, y)    
#    print('Optimal number of features: %d' % greedy_selector.n_features_)
#    
#    X_inter = X_inter.loc[:, greedy_selector.support_]
#    
#    plt.figure()
#    plt.xlabel("Number of features selected")
#    plt.ylabel("Cross validation score (nb of correct classifications)")
#    plt.plot(range(1, len(greedy_selector.grid_scores_) + 1), greedy_selector.grid_scores_)
#    plt.show()
    
    presentSection("Feature Scaling")
    
    presentStepTo("apply feature scaling")
    
    X_std = X.copy()
    
    stdsc = StandardScaler(copy=True, with_mean=True, with_std=True) #must use standard scaler based on params from training, these  same params are appplied to the test data after.
    X_std[['Age','Fare']] = stdsc.fit_transform(X_std[['Age','Fare']]) #fit and transform, to get params from X
    
    X = X_std.copy()      
         
    print(X.head())
    
    presentStepTo("split data into training and test sets")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
    
    printf("Observations in X_train: %d\n Observations in X_test: %d\nObservations in y_train: %d\nObservations in y_test: %d\n", len(X_train), len(X_test), len(y_train), len(y_test))
    
#    presentSection("Dimensionality Reduction")
    
#    presentStepTo("apply principal component analysis")
#    
#    pipe_lr = Pipeline([('pca', PCA()),
#                        ('clf', LogisticRegression(penalty='l2', 
#                                                   random_state=1,
#                                                   verbose=1))])
#    
#    #need to fix validation curve function, it should simply plot curve
#    comp_range = range(1,12,1)
#    train_scores, test_scores = validation_curve(estimator=pipe_lr, 
#                                                 X=X_train, 
#                                                 y=y_train,
#                                                 param_name='pca__n_components',
#                                                 param_range=comp_range,
#                                                 cv=10)
#    train_mean = np.mean(train_scores, axis=1)
#    train_std = np.std(train_scores, axis=1)
#    test_mean = np.mean(test_scores, axis=1)
#    test_std = np.std(test_scores, axis=1)
#    title = "Validation Curve for Logistic Regression"
#    param_title = "Principal Components Analysis"
#    
#    plot_validation_curve(title, 
#                          param_title, 
#                          comp_range, 
#                          train_mean, 
#                          train_std, 
#                          test_mean, 
#                          test_std)
#    
#     
#    idx = test_mean.argmax() #an array of the cumulative variance explained ratio, find first max
#    opt_components = comp_range[idx]
    
    presentSection("train models with machine learning algorithms and find best performer")
    
    alg = menu.presentMLMenu()
    
    if (alg == "LR"):    
        clf = LogisticRegression(penalty='l2', random_state=1, verbose=1)
        param_grid = [{'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
               
    if (alg == "SVC"):
        clf = SVC(random_state=1, verbose=1)
#        param_grid = [{'C': [0.0001, 0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']}]
        param_grid = [{'C': [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], 'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'kernel': ['rbf']}]
        
    if (alg == "DT"):
         clf = tree.DecisionTreeClassifier(random_state=0)
         param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7]}];
        
    if (alg == "RF"):
        clf = RandomForestClassifier(criterion='entropy',
                                    n_estimators=10,
                                    random_state=1, 
                                    n_jobs=-1)
        param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7]}]
        
    if (alg == "ANN"):
        clf = MLPClassifier(solver='sgd', 
                           alpha=0.0001, #alpha is regularization term for L2
                           hidden_layer_sizes=(4, 5), #hiddenlayer1 = 5 neurons, hiddenlayer2 = 2 neurons 
                           max_iter=10000,
                           random_state=1)
        #parameters, learning_rate: adaptive, constant, learning_rate_init, activation, alpha, hidden_layer_sizes
        #need to tune one parameter at a time.
        param_grid = [{'hidden_layer_sizes': list(itertools.product(range(1,6), repeat=2))}, {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]}]
#        param_grid = [{'alpha': [0.0001]}]
#        param_grid = [{'learning_rate': ['adaptive', 'constant']}]
            
    if (alg == "KNN"):
        clf = KNeighborsClassifier(n_neighbors=17, weights='uniform', n_jobs=-1)
        param_grid = [{'n_neighbors': [i for i in range(7,25,1)]}]
#        param_grid = [{'weights': ['uniform', 'distance']}]
        

    scores = GridSearchCV(estimator=clf,
                  param_grid=param_grid,
                              scoring='accuracy',
                              cv=10,
                              verbose=1,
                              return_train_score=True,
                              n_jobs=-1)
    scores.fit(X_train, y_train)
    presentStepTo("show grid search cv results")
    print('Exhasutive Grid Search CV results\n')
    scores_df = pd.DataFrame(data=scores.cv_results_)
    print(scores_df)
    clf = scores.best_estimator_               
    print("Optimal Model ", clf)
    print("Accuracy: ", scores.best_score_)
    y_pred = clf.predict(X_test)    
    print (classification_report(y_test, y_pred))
    
    
    presentStepTo("show validation curve")

    #validation curve
    #If the training score and the validation score are both low, 
    #the estimator will be underfitting. If the training score is high 
    #and the validation score is low, the estimator is overfitting 
    #and otherwise it is working very well. 
    
    #validation curve: plots both training score and cross-validation score.  
    #Y-axis: score, x-axis: parameter being tuned.
    train_scores_mean = scores_df['mean_train_score'].values
    train_scores_std = scores_df['std_train_score'].values
    test_scores_mean = scores_df['mean_test_score'].values
    test_scores_std = scores_df['std_test_score'].values
    params = scores_df.iloc[:, 4].tolist()
    param_title = scores_df.columns[4]
    title = "Validation Curve for " + alg

    plot_validation_curve(title, param_title, params, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)
    
    scores = cross_val_score(clf,
                             X_train,
                             y_train,
                             scoring='accuracy',
                             cv=10)
    
    print('\n10 Folds CV for best model, accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
    
    #learning curve:
    #shows validation and training score of an estimator for various number of training samples.
    #Tells us if we benefit from adding more training data and whether the estimator suffers from
    #variance error or a bias error
    #if both validation score and training score converge to a value that is too low with
    #increasing size of the training set, we will not benefit much from more training data. (high bias, may be underfitting)
    #If the training score is much greater than the validation accuracy score for the maximum number of training
    #samples, more data will most likely increase generalization.  Look for the gap. (high variance, may be over fitting)
        
    
    presentStepTo("show learning curve")

    train_sizes, train_scores, test_scores = learning_curve(estimator=clf, 
                                                        X=X_train, 
                                                        y=y_train, 
                                                        train_sizes=np.linspace(0.1, 1.0, 10), #we set train_sizes=np.linspace(0.1, 1.0, 10) to use 10 evenly spaced relative intervals for the training set sizes.
                                                        cv=10, #k=10, set via the cv param
                                                        n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    title = "Learning Curve for " + alg    
    plot_learning_curve(title, train_sizes, train_mean, train_std, test_mean, test_std)
    
    presentStepTo("\nshow ROC curve and calculate Area Under the Curve")

    #ROC - receive operator curve
    fpr, tpr, thresholds = roc_curve(y_test,y_pred)
    plot_roc_curve(fpr, tpr, roc_auc_score(y_test, y_pred))

    presentStepTo("show confusion matrix")
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(confmat)
    plot_confusion_matrix(confmat)
    
    #decision boundary
    
#    plot_decision_boundary(clf, X_test.values.astype(float), y_test.values)

                
    if(alg == 'DT'):    
        dot_data = tree.export_graphviz(clf, class_names=["Did not Survive","Survived"], feature_names=X_test.columns, out_file="myTree.dot", filled=True)
        with open ("myTree.dot") as f:
            dot_graph = f.read()   
        graphviz.Source(dot_graph)
    


    
    
    
   

