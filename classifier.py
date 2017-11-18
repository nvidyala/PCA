import pca
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

y = []
cov1 = []
cov2 = []
cov3 = []
def classifier():

	iris = datasets.load_iris()
	X1,X2,X3 = [np.transpose(X) for X in pca.main()]

	X1_train, X1_test, y1_train, y1_test = train_test_split(X1,iris.target,test_size=0.3)
	X2_train, X2_test, y2_train, y2_test = train_test_split(X2,iris.target,test_size=0.3)
	X3_train, X3_test, y3_train, y3_test = train_test_split(X3,iris.target,test_size=0.3)

	clf1 = svm.SVC(kernel='linear',C=1)
	clf2 = svm.SVC(kernel='linear',C=1)
	clf3 = svm.SVC(kernel='linear',C=1)

	clf1.fit(X1_train,y1_train)
	clf2.fit(X2_train,y2_train)
	clf3.fit(X3_train,y3_train)

	cov = (clf1.score(X1_test,y1_test))
	corr1 = clf2.score(X2_test,y2_test)
	corr2 = clf3.score(X3_test,y3_test)
	corr = (max(clf2.score(X2_test,y2_test),clf3.score(X3_test,y3_test)))
	x = [cov,corr1,corr2]
	y.append(x)
	print(y)
	cov1.append(x[0])
	cov2.append(x[1])
	cov3.append(x[2])
	

def main():

	domain = []
	test = []
	numiter = 500
	for i in range(numiter):
		classifier()
	
	for i in range(numiter):
		domain.append(i)
		test.append(0.0)



	plt.plot(domain,cov1,'ro',markersize=1)
	plt.plot(domain,cov2,'go',markersize=1)
	plt.plot(domain,cov3,'bo',markersize=1)
	plt.plot(domain,test,'wo')
	plt.show()
	
if __name__ == "__main__":
	main()
