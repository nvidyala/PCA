import pca
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import numpy as np

iris = datasets.load_iris()
X1,X2,X3 = [np.transpose(X) for X in pca.main()]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1,iris.target,test_size=0.3)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,iris.target,test_size=0.3)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,iris.target,test_size=0.3)

clf = svm.SVC(kernel='linear',C=1)

clf.fit(X1_train,y1_train)
print(clf.score(X1_test,y1_test))
clf.fit(X2_train,y2_train)
print(clf.score(X2_test,y2_test))
clf.fit(X3_train,y3_train)
print(clf.score(X3_test,y3_test))
