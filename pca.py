# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 14:32:00 2017

@author: hecto
"""

#covariance matrix
cov_mat = np.cov(X_train_std.T) #.T transpose index and columns
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) #decompose into eigen values and eigen vectors
printf('\n Eigenvalues: %s\n', eigen_vals)

#create array of variance explained ratios of the eigenvalues
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp) #Using cumsum function, we can calculate the cumulative sum of explained variances

#plot on graph to show how much each principal component accounts for the variance

#bar shows each the variance explained ratio of each component
plt.bar(range(1, len(var_exp) + 1), 
        var_exp, 
        alpha=0.5, 
        align='center',
        label='individual explained variance')

#step shows the cumulative explained variance
plt.step(range(1, len(cum_var_exp) + 1), 
         cum_var_exp, 
         where='mid',
         label='cumulative explained variance')

plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.show()

#sort the eigenpairs by decreasing order of the eigenvalues:

eigen_pairs =[(np.abs(eigen_vals[i]), eigen_vecs[:,i]) for i in range(len(eigen_vals))]
eigen_pairs.sort(reverse=True)

#next choose top k PCAs based on the sort above.  The decision a is trade-off between 
#computational efficiency and the performance of the classifier.
#This creates the projection matrix:
    
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis])) #first value in eigen_pair [i], selects the eigen value.  Second value, eigen_pair[i][i] selects the associated eigenvector (PCA). It creates, an array of [[PCA1, PCA2], [PCA1, PCA2]...]

printf('Matrix W:\n'); print w;

#Using the projection matrix, we can now transform a sample x (represented as 
#1×13-dimensional row vector) onto the PCA subspace obtaining x' , a now 
#k-dimensional sample vector consisting of the new k features.  In the case below, k = 2,
#we transform the original training set: 891 X 12 into 891 X 2

X_train_pca = X_train_std.dot(w)


#scikit implementation of pca
#pca = PCA(n_components=None)
#X_train_pca = pca.fit_transform(X_train_std)
#pca.explained_variance_ratio_

