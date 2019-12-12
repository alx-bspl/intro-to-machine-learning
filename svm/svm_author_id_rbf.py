#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from collections import Counter
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

scale = 1 / 100
features_train = features_train[:int(len(features_train)*scale)]
labels_train = labels_train[:int(len(labels_train)*scale)]

#########################################################
### your code goes here ###
def predict(C, scale = 1):
    clf = SVC(kernel='rbf', C=C, gamma='auto')
    print(f"Using {clf}")

    t0 = time()
    clf.fit(features_train, labels_train)
    print(f"Training time: {round(time()-t0, 3)}")

    t0 = time()
    labels_pred = clf.predict(features_test)
    print(f"Prediction time: {round(time()-t0, 3)}")

    acc = accuracy_score(labels_test, labels_pred)
    print(f"Accuracy: {acc}")

    for i in (10, 26, 50):
        print(f"#{i} - {labels_pred[i]} ({'Sara' if labels_pred[i] == 0 else 'Chris'})")

    print(f"Counts - {Counter(labels_pred)}")


for C in reversed((1, 10, 100, 1000, 10000)):
    predict(C)
    print()
#########################################################


