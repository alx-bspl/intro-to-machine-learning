#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
clf = LinearSVC()
print(f"Using {clf}")

t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0, 3)}")  # ~0.495s

t0 = time()
labels_pred = clf.predict(features_test)
print(f"Prediction time: {round(time()-t0, 3)}")  # ~0.015

acc = accuracy_score(labels_test, labels_pred)
print(f"Accuracy: {acc}")  # ~0.990
#########################################################


