#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
clf = GaussianNB()
print(f"Using {clf}")

t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0, 3)}")  # ~2.6s

t0 = time()
labels_pred = clf.predict(features_test)
print(f"Prediction time: {round(time()-t0, 3)}")  # 0.5s

acc = accuracy_score(labels_test, labels_pred)
print(f"Accuracy: {acc}")  # ~0.973

#########################################################
