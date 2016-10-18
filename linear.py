from readFile import readDataSet
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from pca import do
data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]



# clf = KernelRidge(alpha=1)
clf = SVR(C=1.0, epsilon=0)
clf.fit(X, y)
print clf.predict(X)
print y
print clf.score(X,y)
print clf.get_params(deep=True)