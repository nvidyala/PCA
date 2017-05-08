import matplotlib.pyplot as plt
import scipy.stats as stat
import numpy as np
from numpy import linalg as la
from sklearn import datasets


def selection(data):

	attributes = []
	index = []
	attributes = data

	#variance matrix for variables
	var = [0 for i in range(len(attributes))]

	for i in range(len(attributes)):

		mean = sum(attributes[i])/len(attributes[i])

		for value in attributes[i]:
			var[i] += ((value - mean)**2)/len(attributes[i])

	#outlier removal
	low = np.percentile(var,25)
	high = np.percentile(var,75)

	outlier_range = 1.5*(high-low)
	low = low - outlier_range
	high = high - outlier_range

	for value in var:
		if (value < low) or (value > high):
			index.append(var.index(value))
			var.remove(value)

	return index

def cov_matrix(data,n):

	cov = []
	data_t = np.transpose(data)
	cov = np.dot(data,data_t)
	cov = cov/(n-1)


	cov_eigval = la.eig(cov)[0]
	cov_eigvec = la.eig(cov)[1]

	cov_eigvec = np.transpose(cov_eigvec)

	#eig_test = [[-.67787, -.73517],[-.73517, 0.67787]]

	final_data = np.dot(cov_eigvec,data)
	
	select = selection(final_data)

	for index in select:
		np.delete(final_data,index)

	return final_data



def correlation_matrix(data,n):

	pearson = []
	spearman = []

	for i in range(0,len(data)):
		for j in range(0,len(data)):
			pearson.append(stat.pearsonr(data[i],data[j])[0])
			spearman.append(stat.spearmanr(data[i],data[j])[0])

	pearson,spearman = np.reshape(pearson,(-1,len(data))),np.reshape(spearman,(-1,len(data)))

	pearson_eigval = la.eig(pearson)[0]
	pearson_eigvec = la.eig(pearson)[1]

	spearman_eigval = la.eig(spearman)[0]
	spearman_eigvec = la.eig(spearman)[1]

	pearson_eigvec = np.transpose(pearson_eigvec)
	spearman_eigvec = np.transpose(spearman_eigvec)

	pearson_data = np.dot(pearson_eigvec,data)
	spearman_data = np.dot(spearman_eigvec,data)

	select_p = (selection(pearson_data))
	select_s = (selection(spearman_data))

	#data reduction
	for index in select_p:
		pearson_data = np.delete(pearson_data,index,0)

	for index in select_s:
		spearman_data = np.delete(spearman_data,index,0)

	return pearson_data,spearman_data

def main():

	iris = datasets.load_iris()
	iris_X = iris.data
	print(iris)
	X = [[] for i in range(0,len(iris.feature_names))]
	for i in range(0,len(iris.feature_names)):
		for sample in iris_X:
			X[i].append(sample[i])

	target_data1 = X

	for feature in X:
		mean = sum(feature)/len(feature)

		for value in feature:
			value = value-mean

	target_data2 = X

	cov_data = cov_matrix(target_data2,150)
	pearson_data,spearman_data = correlation_matrix(target_data1,150)

	return cov_data,pearson_data,spearman_data



if __name__ == "__main__":
	main()

