from readFile import readDataSet
from sklearn.externals import joblib
import numpy as np
data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

clf = joblib.load('SGD_Regressor_20k.pkl')
# clf = joblib.load('PCA_20k.pkl')
# best_SVR_20k.pkl 
#print clf
pred = clf.predict(X)
print pred
print clf.score(X,y)
print np.sum((pred-y)**2)/len(y)

# from readFile import readDataSet
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.externals import joblib

# data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
# X = data[:,1:91]
# y = data[:,0]

# clf = KernelRidge(alpha = 1e-3)
# clf.fit(X, y)
# joblib.dump(clf, "linear_KRR_20k.pkl")

# # clf = joblib.load('filename.pkl') 
# print clf.predict(X)
