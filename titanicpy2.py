# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:11:02 2017

@author: hecto
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import sys
import scipy.stats as stats


def printf(format, *args):
    sys.stdout.write(format % args)

#exploring the data
print "..............................................................................."
print "Exploring and Cleaning the Data"
print "...............................................................................\n"

print "<Loading the Titantic Challenge training and test .csv files>\n"
df_train = pd.read_csv("C:\\Users\\hecto\\Documents\\Programming\\Kaggle\\Titantic Challenge\\train.csv")
df_submit_test = pd.read_csv("C:\\Users\\hecto\\Documents\\Programming\\Kaggle\\Titantic Challenge\\test.csv")



print "Identified the Survived Feature as the Target label in the data set....."
print "<Splitting the Y Target label from the data set and storing in seperate array y>\n"
print "<X_train stores the training data, X_test stores the test data>\n"
y = df_train["Survived"];
X_train = df_train
X_test = df_submit_test
del X_train["Survived"];

print "Remove Unneeded features from the dataset...\n"
print "<Removing feature PassengerID>"
print "PassengerID is an arbitrarily assigned number to each passenger\n"
print "<Removing feature Ticket Number>"
print "Ticket Number Ticket data does not seem to provide any information \nthat we can't get from the Passenger Class and Fare features.\n"
print "<Removing Feature Cabin>"
print "Cabin is missing data too much data.  In cabin (687 of 892 rows are \nmissing this data for this feature), besides the location of the cabin \na passenger was staying at does not equate to the location of the passenger \nwhen the Titanic was sinking.\n"

#Remove Unneeded columns
#remove PassengerID, Ticket, Cabin from dataframe.  Ticket data does not provide any
#information that we can't get from the pclass and fare.  There is too much missing data
#in cabin (687 missing fields of 892), besides the location of the cabin a passenger was staying at does not equate to
#the location of the passenger when the Titanic was sinking.  Note, we could have removed
#rows with missing values with the method: df.dropna().  But this would not be the right move,
#with so much cabin information missing. Names will not matter.
del X_train["PassengerId"], X_train["Ticket"], X_train["Cabin"], X_train["Name"]
del X_test["PassengerId"], X_test["Ticket"], X_test["Cabin"], X_test["Name"]

raw_input("Press Enter to continue...\n")

print "Search data for missing values: NaNs (not a number), blanks, and nulls...\n"
print "Training data results for null values: \n"
print X_train.isnull().sum(), "\n"
print "Test data results for missing values: \n"
print X_test.isnull().sum(), "\n"
#interpolation
#For age, fill in missing values with with mean Age rounded to 1 decimal place, 
#since some data does reflect half years.  This process is called "mean imputation."
print "Interpolation, imputing mean, median and mode...\n"
print "For Age, imputing the median age is appropriate since the value measures..."
print "Median Age in training set (rounded to 1 decimal place): ", round(X_train['Age'].median(),1)
print "<Using this median to impute into missing values for the Age feature in training and test set>\n"

from sklearn import linear_model
#Univariate linear regression to predict missing fare values

#f_train = X_train[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age']]
#f_train = f_train.dropna()
#f_y_train = f_train['Age']
#del f_train['Age']
#
#
#for col in ['Pclass','Sex', 'Embarked']:
#    temp1 = pd.get_dummies(f_train[col]) 
#    del f_train[col]
#    f_train = pd.concat([f_train, temp1], axis=1)
#
#f_train[["SibSp","Parch"]] = f_train[["SibSp","Parch"]].astype('float64')    



#cutoff = int(f_train.shape[0] * .7)
#f_test = pd.DataFrame(f_train[cutoff:]) #because it was made into series automatically, needs to be a dataframe for linear reg
#f_train = pd.DataFrame(f_train[:cutoff])
#f_y_test = pd.DataFrame(f_y_train[cutoff:])
#f_y_train = pd.DataFrame(f_y_train[:cutoff])
#
#regr = linear_model.LinearRegression()
#regr.fit(f_train, f_y_train)

#print "Coefficients: ", regr.coef_
#printf ('Variance score: %.2f' % regr.score(f_test, f_y_test))
#
## Plot outputs
#plt.scatter(f_test, f_y_test,  color='black')
#plt.plot(f_test, regr.predict(f_test), color='blue',
#         linewidth=3)
#
#plt.xticks(())
#plt.yticks(())
#
#plt.show()


X_train['Age'] = X_train['Age'].fillna(X_train.median()['Age'])
X_train = X_train.round({'Age': 1})
X_test['Age'] = X_test['Age'].fillna(X_test.median()['Age'])
X_test = X_test.round({'Age': 1})

raw_input("Press Enter to continue...\n")

print "For Fare, mean imputation is appropriate since the value measures..."
print "Mean Fare by passenger class in training set(rounded to two decimal places): " 

class_1 = X_train[(X_train["Pclass"] == 1)]
class_2 = X_train[(X_train["Pclass"] == 2)]
class_3 = X_train[(X_train["Pclass"] == 3)]
fare1 = round(class_1['Fare'].mean(), 2) 
fare2 = round(class_2['Fare'].mean(), 2)         
fare3 = round(class_3['Fare'].mean(), 2)  
                  
print "Class 1: ", fare1 , "Class 2: ", fare2, "Class 3: ", fare3
print "<Using this mean to impute into missing values for the Fare feature in training and test set>\n"


X_train[(X_train["Pclass"] == 1)] = X_train['Fare'].fillna(fare1)
X_train[(X_train["Pclass"] == 2)] = X_train['Fare'].fillna(fare2)
X_train[(X_train["Pclass"] == 3)] = X_train['Fare'].fillna(fare3)   

X_test[(X_train["Pclass"] == 1)] = X_test['Fare'].fillna(fare1)
X_test[(X_train["Pclass"] == 2)] = X_test['Fare'].fillna(fare2)
X_test[(X_train["Pclass"] == 3)] = X_test['Fare'].fillna(fare3)     

print X_train
raw_input("Press Enter to continue...\n")
print X_test
#X_test['Fare'] = X_test['Fare'].fillna(X_test.mean()['Fare'])
#X_test = X_test.round({'Fare': 2})

raw_input("Press Enter to continue...\n")

#embark last column left with missing values, very minimal, will use most frequent 
#occurrence to fill in blanks.
print "For Embarked, mode imputation is appropriate since it is a categorical feature..."
print "Mode of Embarked in training set: ", X_train['Embarked'].value_counts().index[0]
print "<Using this meode to impute into missing values for the Embarked feature in training and test set>/n"

mode = X_train["Embarked"].value_counts().index[0]
X_train['Embarked'] = X_train['Embarked'].fillna(mode)

raw_input("Press Enter to continue...\n")
#no more missing values verified:
print "Verifying that all missing values cleaned up..."
print "Training data results for null values: \n"
print X_train.isnull().sum()
print "Test data results for null values: \n"
print X_test.isnull().sum()


#additional note:  Mean imputation can be achieved through the scikit learn kit.
#However, in this case, mean imputation was only needed for one row and the code was much 
#shorter to implement through pandas. Sklearn example:

#from sklearn.preprocessing import Imputer#
#imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
#imr = imr.fit(df)
#imputed_data = imr.transform(df.values)
#imputed_data

#Here, we replaced each NaN value by the corresponding mean, which is separately 
#calculated for each feature column. If we changed the setting axis=0 to axis=1, 
#we'd calculate the row means. Other options for the strategy parameter are median 
#or most_frequent, where the latter replaces the missing values by the most frequent values. 
#This is useful for imputing categorical feature values.

#Now that we have decided which features we will keep and have imputed where we can,
#we could drop all rows with a nan value, but there are none.  

#X_train = X_train.dropna()

#If we wanted to select the column it would be
#df.dropna(subset=['column_name'])

#data transformation (feature standardization)
#The result of standardization (or Z-score normalization) is that the features will 
#be rescaled so that they’ll have the properties of a standard normal distribution with:
#mean=0 and sd=1.  This step is necessary because we have very different scales for each
#feature.  This will help our algorithm run faster at training time.  We use the z-score
#standardization approach (mean=0 and sd=1).  But the min-max scaling approach which 
#places the data is scaled to a fixed range - usually 0 to 1. The cost of having this 
#bounded range - in contrast to standardization - is that we will end up with smaller 
#standard deviations, which can suppress the effect of outliers.  A popular application 
#is image processing, where pixel intensities have to be normalized to fit within a 
#certain range (i.e., 0 to 255 for the RGB color range). Also, typical neural network 
#algorithm require data that on a 0-1 scale. 

print "Visualizng and Describing the data\n"
print "<Converting Categorial features: PClass, Sex, Embarked,and (target) Survived into dtype=category for descriptive statistics>" 


for col in ['Pclass','Sex', 'Embarked']:
    temp1 = pd.get_dummies(X_train[col]) 
    temp2 = pd.get_dummies(X_test[col]) 
    del X_train[col], X_test[col]
    X_train = pd.concat([X_train, temp1], axis=1)
    X_test = pd.concat([X_test, temp2], axis=1)

X_train.rename(columns={1:'Pclass_1', 2:'Pclass_2', 3:'Pclass_3', 'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southamption'}, inplace=True)
X_test.rename(columns={1:'Pclass_1', 2: 'Pclass_2', 3:'Pclass_3', 'C':'Cherbourg', 'Q':'Queenstown', 'S':'Southamption'}, inplace=True)

print "Convert integers to floating numbers, this is necessary for scilean Random Forest Algorithm"

X_train[["SibSp","Parch"]] = X_train[["SibSp","Parch"]].astype('float64')    
X_test[["SibSp","Parch"]] = X_test[["SibSp","Parch"]].astype('float64') 

print X_train.head(10)
print X_test.head(10)
 
descriptive_stats = X_train.describe()


print descriptive_stats.round(4)

print "Encode categorical variables PClass, Sex, Embarked...\n"


print "Data Transformation, last step to clean data - feature scaling applied to the data set"
X_train_std = X_train
X_test_std = X_test

stdsc = StandardScaler(copy=True, with_mean=True, with_std=True) #must use standard scaler based on params from training, these  same params are appplied to the test data after.
X_train_std[['Age','Fare']] = stdsc.fit_transform(X_train[['Age','Fare']]) #fit and transform, to get params from X_train
X_test_std[['Age','Fare']] = stdsc.transform(X_test[['Age','Fare']])#only transform, because we will use the params from X_train


raw_input("Press Enter to continue...\n")

print "..............................................................................."
print "Split Data into Training, Cross-Validation and Test sets"
print "...............................................................................\n"

#at this point we split the X_train into train, cross-validation, test sets, with 50% for train
#train_percentage = X_train_std.shape[0]/float(X_train_std.shape[0] + X_test_std.shape[0])

def percentage_breakdown(num, den):
    pb = round(num/float(den), 4)
    return pb

rows_Xtr = X_train_std.shape[0]
rows_Xtest = X_test_std.shape[0]  
total_rows = rows_Xtr + rows_Xtest  
    
train_percentage = percentage_breakdown(rows_Xtr, total_rows)
test_percentage = percentage_breakdown(rows_Xtest, total_rows)
target_rows = (total_rows) * (train_percentage - .5)
cross_val_percentage = percentage_breakdown(target_rows, total_rows)

printf("X_train_std # of samples: %d \n", rows_Xtr)
printf("X_test_std # of samples: %d\n\n", rows_Xtest)
print "Percentage breakdown between Training and Test Sets as given by Kaggle Titantic data set...\n"
printf("train_percentage: %04.2f%% \n", train_percentage * 100)
printf("test_percentage: %04.2f%% \n", test_percentage * 100)
print ""
print "Breakdown for splitting into Training, Cross-Validation and Test sets...\n"
#note: Can't split from test set because it does not have any target values.

cutoff =  int(round(rows_Xtr - target_rows)) - 1

X_cv_std = X_train_std[cutoff:]
X_train_std = X_train_std[0:cutoff]
y_cv = y[cutoff:]
y_train = y[0:cutoff]

#percentage breakdown
#rows_Xtr = X_train_std.shape[0]
rows_Xtr = X_train_std.shape[0]
rows_Xtest = X_test_std.shape[0]
rows_Xcv = X_cv_std.shape[0]

#rows_Xtest = X_test_std.shape[0]

printf("Rows in training: %d................Percentage: %04.2f%%\n", rows_Xtr, percentage_breakdown(rows_Xtr, rows_Xtr + rows_Xcv + rows_Xtest) * 100)
printf("Rows in cross-validation: %d........Percentage: %04.2f%%\n", rows_Xcv,  percentage_breakdown(rows_Xcv, rows_Xtr + rows_Xcv + rows_Xtest) * 100)
printf("Rows in test: %d....................Percentage: %04.2f%%\n\n", rows_Xtest,  percentage_breakdown(rows_Xtest, rows_Xtr + rows_Xcv + rows_Xtest) * 100)
 
#Again, it is also important to highlight that we fit the StandardScaler only once
#on the training data and use those parameters to transform the test set or any new 
#data point.  With the StandardScaler, centering and scaling happen independently on each feature by computing 
#the relevant statistics on the samples in the training set.

#alternative cross-validation splitting: 
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


#Classification, y_train
#contains categorical data that is nominal not ordinal.  That is, it can't be sorted 
#ordered in any way.  It is, did this person survive, yes or no.
#A note on encoding class labels.
#Many machine learning libraries require that class labels are encoded as integer values. 
#Although most estimators for classification in scikit-learn convert class labels to 
#integers internally, it is considered good practice to provide class labels as integer 
#arrays to avoid technical glitches.  We need to remember that class labels are not 
#ordinal, and it doesn't matter which integer number we assign to a particular string-label.
#While in this case, this is not a concern, since our y labels are already in int form,
#here is an example of quickly mapping strings into integer classes.

#import numpy as np#
#class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
#class_mapping
#{'class1': 0, 'class2': 1}
#df['classlabel'] = df['classlabel'].map(class_mapping) (pg.105, a lot of scenarios discussed)


print "..............................................................................."
print "Covariance Matrix, Decompose into Eigenvectors and values,\n"
print "and the Variance Explained Ratio"
print "...............................................................................\n"

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=cmap(idx),
                    marker=markers[idx], 
                    label=cl)
        
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

pca = PCA(n_components=6)
lr = LogisticRegression()
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_cv_std)
lr.fit(X_train_pca, y_train)
#plot_decision_regions(X_train_pca, y_train, classifier=lr)
#plt.xlabel('PC1')
#plt.ylabel('PC2')
#plt.legend(loc='lower left')
#plt.show()

printf ("score: %04.2f", lr.score(X_test_pca, y_cv))





print "..............................................................................."
print "Choosing a Learning Algorithm"
print "...............................................................................\n"

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#note: the perceptron algorithm never converges on datasets that aren't perfectly 
#linearly separable, which is why the use of the perceptron algorithm is typically 
#not recommended in practice

print "Perceptron"
learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
misclass_total = X_cv_std.shape[0] #assume it has everything wrong to begin with
 
for rate in learning_rates:
    ppn = Perceptron(n_iter=10000, eta0=rate, random_state=0) #epochs,learning rate, random_state = the initial shuffling of the training dataset after each epoch
    ppn.fit(X_train_std, y_train)
    y_pred = ppn.predict(X_cv_std)
    current_misclass = (y_cv != y_pred).sum()
    if(current_misclass <= misclass_total):
        misclass_total = current_misclass
        learning_rate = rate

printf('Selected Learning_rate: %04.6f\n' % learning_rate)
printf('Misclassified samples: %d\n' % misclass_total) 

misclass_error = misclass_total/float( y_cv.shape[0])
printf("Misclassification error: %04.2f\n", misclass_error)

accuarcy_score = accuracy_score(y_cv, y_pred)
print('Accuracy: %.2f\n' % accuarcy_score)

#def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):
#	# setup marker generator and color map
#	markers = ('s', 'x', 'o', '^', 'v')	
#	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#	cmap = ListedColormap(colors[:len(np.unique(y))]) #creates color map from the above marker and color attributes from matplotlib.  np.unique(y) returns the sorted unique elements of an array.
#	
#	# plot the decision surface
#	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # max and min of feature x_1 and x_2 (below) max() and min() are methods for pandas object
#	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1 
#	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
#	np.arange(x2_min, x2_max, resolution)) #a pair of grid arrays xx1 and xx2 via the NumPy meshgrid function.  The meshgrid function returns coordinate matrices from coordinate vectors.  The arange function returns evenly spaced values within a given interval.
#	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T) #classifier is the perceptron, we use the predict method (aka the output function) to predict the class labels z of the corresponding grid points.  The ravel function returns a flattened array.
#	Z = Z.reshape(xx1.shape)
#	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
#	plt.xlim(xx1.min(), xx1.max())
#	plt.ylim(xx2.min(), xx2.max())
#
#	# plot class samples
#	for idx, cl in enumerate(np.unique(y)):
#		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
#			alpha=0.8, c=cmap(idx),
#			marker=markers[idx], label=cl)
#    #highlight test samples
#        if test_idx:
#            X_test, y_test = X[test_idx, :], y[test_idx]
#            plt.scatter(X_test[:, 0], X_test[:, 1], c='',
#                    alpha=1.0, linewidth=1, marker='o',
#                    s=55, label='test set')

#X_combined_std = np.vstack((X_train_std, X_cv_std))
#y_combined = np.hstack((y_train, y_cv))
#plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105,150))
#plt.xlabel('petal length [standardized]')
#plt.ylabel('petal width [standardized]')
#plt.legend(loc='upper left')
#plt.show()        
        
print "Logistic Regression: linear"
from sklearn.linear_model import LogisticRegression

#C is directly related to the regularization parameter lambda, which is its 
#inverse: 1/lambda.  Decreasing the value of the inverse regularization parameter
#C means that we are increasing the regularization strength

lr_misclass_total = X_cv_std.shape[0]
weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0, penalty='l2')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_) #weights stored given each run params
    params.append(10**c) #c param
    y_pred = lr.predict(X_cv_std)
    lr_current_misclass = (y_cv != y_pred).sum()
    if(current_misclass <= misclass_total):
        lr_misclass_total = lr_current_misclass
        opt_weights = lr.coef_
        opt_param = 10**c

printf('Selected C: %04.6f\n' % opt_param)
printf('Misclassified samples: %d\n' % lr_misclass_total) 

misclass_error = misclass_total/float( y_cv.shape[0])
printf("Misclassification error: %04.2f\n", misclass_error)

accuarcy_score = accuracy_score(y_cv, y_pred)
print('Accuracy: %04.2f\n' % accuarcy_score)

print "Support Vector Machine: linear"
#the variable C, we can then control the penalty for misclassification. Large values of C correspond to large error penalties whereas we are less strict about misclassification errors if we choose smaller values for C
#log vs svm: logistic regression models can be easily updated, which is attractive when working with streaming data.    
#from sklearn.svm import SVC
#
#svm_misclass_total = X_cv_std.shape[0]
#weights, params = [], []
#
#for c in np.arange(-5, 5):
#    svm = SVC(kernel='linear', C=10**c, random_state=0)
#    svm.fit(X_train_std, y_train)
#    y_pred = svm.predict(X_cv_std)
#    current_misclass = (y_cv != y_pred).sum()
#    if(current_misclass <= misclass_total):
#        svm_misclass_total = current_misclass
#        opt_weights = svm.coef_
#        opt_param = 10**c
#        
#printf('Selected C: %04.6f\n' % opt_param)
#printf('Misclassified samples: %d\n' % svm_misclass_total) 
#
#misclass_error = svm_misclass_total/float( y_cv.shape[0])
#printf("Misclassification error: %04.2f\n", misclass_error)
#
#accuarcy_score = accuracy_score(y_cv, y_pred)
#print('Accuracy: %.2f\n' % accuarcy_score)

print "Kernal Support Vector Machine: non-linear"
#Using the kernel trick to find separating hyperplanes in higher dimensional space
#To solve a nonlinear problem using an SVM, we transform the training data onto a 
#higher dimensional feature space via a mapping function ()and train a linear 
#SVM model to classify the data in this new feature space. Then we can use the same 
#mapping function ()

#check to see if there is a pattern in the misclassified examples...
#the term kernel can be interpreted as a similarity function between a pair of samples.
#simply change the kernal param from linear to 'rbf' and use a gamma param 
#(understood as a cutoff). The lower gamma is the sofer the decision boundary,
#the higher it is the harder the decision boundary.

#ksvm_misclass_total = X_cv_std.shape[0]
#weights, params = [], []
#
#for c in np.arange(-5, 5):
#    ksvm = SVC(kernel='rbf', C=10**c, random_state=0, gamma=1.0)
#    ksvm.fit(X_train_std, y_train)
#    y_pred = ksvm.predict(X_cv_std)
#    current_misclass = (y_cv != y_pred).sum()
#    if(current_misclass <= misclass_total):
#        ksvm_misclass_total = current_misclass
#        #opt_weights = svm.coef_
#        opt_param = 10**c
#
#printf('Selected C: %04.6f\n' % opt_param)
#printf('Misclassified samples: %d\n' % ksvm_misclass_total) 
#
#misclass_error = ksvm_misclass_total/float(y_cv.shape[0])
#printf("Misclassification error: %04.2f\n", misclass_error)
#
#accuarcy_score = accuracy_score(y_cv, y_pred)
#print('Accuracy: %.2f\n' % accuarcy_score)

#plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105,150))
#plt.xlabel('petal length [standardized]')
#plt.ylabel('petal width [standardized]')
#plt.legend(loc='upper left')
#plt.show()

#lr = LogisticRegression(C=1000.0, random_state=0) 
#lr.fit(X_train_std, y_train)
#y_pred = lr.predict(X_cv_std)

#predict the class-membership probability of the samples via
#the predict_proba method

print "Decision Tree"
from sklearn.tree import DecisionTreeClassifier
#does not actually need to be feature scaled
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train_std, y_train)
X_combined = np.vstack((X_train_std, X_cv_std))
y_combined = np.hstack((y_train, y_cv))

accuarcy_score = tree.score(X_cv_std, y_cv)
print('Accuracy: %.2f\n' % accuarcy_score)

print "Random Forest"

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
n_estimators=10, random_state=1, n_jobs=2) #jobs indicates how many cores to use.
forest.fit(X_train_std, y_train)

accuarcy_score = tree.score(X_cv_std, y_cv)
print('Accuracy: %.2f\n' % accuarcy_score)


print "K-Nearest Neighbours using an Euclidean distance metric"
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2,
metric='minkowski') #The 'minkowski' distance that we used in the previous code is just a generalization of the Euclidean and Manhattan distance
knn.fit(X_train_std, y_train)