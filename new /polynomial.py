from readFile import readDataSet
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

data, nrows, ncols = readDataSet("YearPredictionMSD100.txt")
X = data[:,1:91]
y = data[:,0]


clf = joblib.load('PCA_100k_4.pkl') 
X = clf.transform(X)
X = StandardScaler().fit_transform(X)
print "pca X: " ,X, len(X),len(X[0])
X =  PolynomialFeatures(degree=6).fit_transform(X)
print "poly features: ",X, len(X),len(X[0])
# clf = Ridge(alpha = 0.001,max_iter=-1,fit_intercept=True)

params = {'alpha':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,0.0001,0.001,0.01,0.1,1,10,100,1000]}
clf = GridSearchCV(Ridge(max_iter=-1), params,  cv = 5,verbose=True)
clf.fit(X,y)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
print clf.cv_results_
print clf.cv_results_['mean_test_score']
# # print "coef:", clf.coef_
# # print clf.n_iter_
# y_pred = clf.predict(X)
# print "y:", y
# print "y_pred:", y_pred
# print (np.sum((y_pred - y)** 2)/len(X))
# print clf.score(X,y)
# # print clf.get_params()

# joblib.dump(clf, "poly_Ridge_20k.pkl")
data, nrows, ncols = readDataSet("YearPredictionMSDTest10.txt")
X = data[:,1:91]
y = data[:,0]
from sklearn.decomposition import PCA

clfp = PCA(n_components = 4)
X = clfp.fit_transform(X) 

X = StandardScaler().fit_transform(X)
X =  PolynomialFeatures(degree=6).fit_transform(X)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)



# clf = joblib.load('filename.pkl') 
