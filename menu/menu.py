# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:36:33 2017

@author: hecto
"""

algorithms = [
        ("LR", "logistic regression"), 
        ("SVC", "support vector classifier"),
        ("DT", "decision tree"),
        ("RF", "random forest"),
        ("ANN", "artificial neural network"),
        ("KNN", "k-nearest neighbours")
        ]

crossValidations = [
        ("GS", "exhaustive grid search and hyperparameter tuning "),
        ("X", "no hyperparameter tuning")
        ]


def presentListAsMenu(menuList):
    for item in menuList:
        print(item[0] + " -- " + item[1])

def checkSelectionIsIn(listToCheck, selection): 
    for item in listToCheck:
        if (selection == item[0]):
            return True
    print("Invalid selection, please check your input")
    return False
 
def presentAlgorithms():
    isAlgorithmSelected = False
    while (isAlgorithmSelected == False):
        print("Select Machine Learning Algorithm to train\n")
        presentListAsMenu(algorithms)
        algorithmSelected = input("Select Algorithm: ")
        isAlgorithmSelected = checkSelectionIsIn(algorithms, algorithmSelected)
    return algorithmSelected
    
  
def presentCrossValidations():
    isCrossValidationSelected = False
    while (isCrossValidationSelected == False):
        print("Select Cross Validation Technique: \n")
        presentListAsMenu(crossValidations)
        crossValidationSelected = input("Cross Validation technique: ")
        isCrossValidationSelected = checkSelectionIsIn(crossValidations, crossValidationSelected)
    return crossValidationSelected
        
def presentMLMenu():
    alg = presentAlgorithms()
    return alg
    
    
   
