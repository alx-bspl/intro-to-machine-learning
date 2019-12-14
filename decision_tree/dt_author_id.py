#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
percentile = 1
features_train, features_test, labels_train, labels_test = preprocess(percentile=percentile)

#########################################################
### your code goes here ###
clf = DecisionTreeClassifier(min_samples_split=40)
print(f"Using {clf}")

t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0, 3)}")  # ~142.91s / ~9.05s

t0 = time()
labels_pred = clf.predict(fe    atures_test)
print(f"Prediction time: {round(time()-t0, 3)}")  # ~0.12s / ~0.003s

acc = accuracy_score(labels_test, labels_pred)
print(f"Accuracy: {acc}")  # 0.9772468714448237 / 0.9670079635949943

features_count = len(features_train[0])
print(f"Features count: {features_count}")  # 3785 / 379
#########################################################


