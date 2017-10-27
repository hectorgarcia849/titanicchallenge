# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 01:05:53 2017

@author: hecto
"""
import matplotlib.pyplot as plt

def plot_learning_curve(title, train_sizes, train_mean, train_std, test_mean, test_std):
    plt.suptitle(title, fontsize=14, fontweight='bold')
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
    plt.ylim([0.0, 1.0])
    plt.show()
    
def plot_validation_curve(title, param_title, param_range, train_mean, train_std, test_mean, test_std):
    plt.suptitle(title, fontsize=14, fontweight='bold')
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
    plt.xlabel(param_title)
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1.0])
    plt.show()
    
def plot_confusion_matrix(confmat):
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