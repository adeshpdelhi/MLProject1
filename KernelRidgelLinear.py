from readFile import readDataSet
from sklearn.kernel_ridge import KernelRidge
from sklearn.externals import joblib

data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

clf = joblib.load('PCA_20k.pkl') 
X = clf.transform(X)
print X
clf = KernelRidge(alpha = 1, kernel="linear")
clf.fit(X, y)
print clf.predict(X)
print clf.get_params()
print clf.score(X,y)

joblib.dump(clf, "KRR_linear_20k.pkl")



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
