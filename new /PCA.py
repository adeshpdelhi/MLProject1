from readFile import readDataSet
from sklearn.externals import joblib
# from sklearn.linear_model import SGDRegressor 
from sklearn.decomposition import PCA
data, nrows, ncols = readDataSet("YearPredictionMSD100.txt")
X = data[:,1:91]
y = data[:,0]

clf = PCA(n_components = 4)
x = clf.fit_transform(X) 
print x, len(x), len(x[0])
clf.fit(X)
print "variance ratio :",clf.explained_variance_ratio_
print "eigenvectors: ", clf.components_
joblib.dump(clf, "PCA_100k_4.pkl")

# clf = joblib.load('filename.pkl') 
