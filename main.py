import random,sys
import numpy as np
import pylab 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# from linear_regression import *
import linear_regression as lr
from linear_regression import *

def readDataSet(file):
	try:
		f = open(file, 'r')
	except IOError:
		print "UNABLE TO OPEN FILE\n"
		return (np.array([]),0,0)
	X = []
	for line in f:
		row = line.split(',')
		if(len(row) <= 1):
			continue
		row[-1] = row[-1].split('\n')[0]
		# Y.append(float(row[-1].split('\n')[0]))
		row = map(float, row[:])
		# row = [1]+ row
		X.append(row)
	f.close()
	return np.array(X), len(X), len(X[0])

### lin dataset
# #linear regression
# file = "lin.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 0.00019 
# print "parameters: ",linear_regression(X, 0,1000, 0)

# #polynomial regression
# file = "lin.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 1e-6
# print "parameters: ",linear_regression(X,1,1000,0)

# # gaussian regression
# file = "lin.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 0.0049
# print "parameters: ",linear_regression(X, 2,1000, 0)

# lr.multiplot(title="MSE vs amount of data; dataSet = lin",xlabel="amount of data",ylabel="MSE",f=file.split('.')[0])

### code to find optimal lambda
# file = "lin.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 0.00019 
# deltas = [0.1,0.5,1,1.5,2,5,10,50]
# for i in deltas:
# 	print "i: ",i
# 	d = kFoldCrossValidation(X,i,0,1)
# 	print "d: ",d

# ## Regularized linear regression
# file = "lin.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 0.00019 
# print "parameters: ",linear_regression(X,0,1000,0.1)


# # # sph dataset
# # linear regression
# file = "sph.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 0.000002552 
# print "parameters: ",linear_regression(X, 0,1000, 0)

# # polynomial regression
# file = "sph.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 1e-11 
# print "parameters: ",linear_regression(X, 1,1000, 0)

# # gaussian regression
# file = "sph.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 5e-4 
# print "parameters: ",linear_regression(X, 2,1000, 0)

# lr.multiplot(title="MSE vs amount of data; dataSet = sph",xlabel="amount of data",ylabel="MSE",f=file.split('.')[0])

## code to find optimal lambda
# file = "sph.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# deltas = [0.1,0.5,1,1.5,2,5,10,50,100,500]
# lr.learningRate = 1e-11 
# for i in deltas:
# 	print "i: ",i
# 	d = kFoldCrossValidation(X,i,1,1)
# 	print "d: ",d

# #Regularized linear regression
# file = "sph.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# lr.learningRate = 1e-11 
# print "parameters: ",linear_regression(X,1,1000,500)

##e)
# file = "iris.data"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# kFoldCrossValidation(X,0.1,0,0)

# file = "seeds_dataset.txt"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# kFoldCrossValidation(X,0.1,0,0)

# file = "AirQualityUCI.csv"
# (X, N, D) = readDataSet(file)
# lr.dataSet = file.split('.')[0]
# kFoldCrossValidation(X,50,0,0)