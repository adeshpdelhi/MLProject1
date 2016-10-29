from sklearn.externals import joblib
from readFile import readDataSet
from sklearn.preprocessing import StandardScaler
import numpy as np
data, nrows, ncols = readDataSet("YearPredictionMSDTest.txt")
X = data[:,1:91]
y = data[:,0]

X = StandardScaler().fit_transform(X)

clf = joblib.load('rbf_SVR_20k_C1000.pkl')
print clf
pred = clf.predict(X)
print pred
print y
print (np.sum((pred - y)** 2)/len(X))
print clf.score(X,y)
