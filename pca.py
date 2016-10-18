from readFile import readDataSet
from sklearn.svm import LinearSVR
import numpy as np
from sklearn.decomposition import PCA

data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]
pca = PCA(n_components=2)
pca.fit(X)
PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_) 
print pca.components_
# print pca.explained_variance_
# print pca.mean_
print pca.n_components_
# print pca.noise_variance_
print pca.components_[1]
rowFeatureVector = pca.components_
X = np.dot(rowFeatureVector,X.transpose())
X = X.transpose()
print len(X)
print X
clf = LinearSVR(C=1.0, epsilon=0, verbose= 1, max_iter=100000)
clf.fit(X, y)
print clf.predict(X)
print y
print clf.score(X,y)
print clf.get_params(deep=True)