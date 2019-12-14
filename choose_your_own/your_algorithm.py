#!/usr/bin/python
import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import pretty_picture
from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast, bumpy_fast, grade_slow, bumpy_slow = (
    [features_train[row][column] for row in range(0, len(features_train)) if labels_train[row] == label]
    for label in (0, 1) for column in (0, 1)
)


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


def k_nearest_neighbors(n_neighbors=5):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    print(f"Using {clf}")

    t0 = time()
    clf.fit(features_train, labels_train)
    print(f"Training time: {round(time() - t0, 3)}")  #

    t0 = time()
    labels_pred = clf.predict(features_test)
    print(f"Prediction time: {round(time() - t0, 3)}")  #

    acc = accuracy_score(labels_test, labels_pred)
    print(f"Accuracy: {acc}")  #

    pretty_picture(clf, features_test, labels_test)


for n in (1, 2, 5, 10, 20, 40):
    k_nearest_neighbors(n_neighbors=n)  # best accuracy: 0.94
