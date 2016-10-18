from readFile import readDataSet
from sklearn.kernel_ridge import KernelRidge

data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

clf = KernelRidge()
print clf.fit(X, y)
