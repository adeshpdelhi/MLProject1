from readFile import readDataSet
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import numpy as np
from sklearn.preprocessing import StandardScaler

data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

clf = joblib.load('PCA_20k.pkl') 
X = clf.transform(X)
print "pca X: " ,X, len(X),len(X[0])
X = StandardScaler().fit_transform(X)

X =  PolynomialFeatures(degree=2).fit_transform(X)
print "poly features: ",X, len(X),len(X[0])
# clf = Ridge(alpha = 0.001,max_iter=100000,normalize="True",fit_intercept=True)
clf = SVR(C=1.0, epsilon=0.2,max_iter=10000,degree = 3, coef0 = 0 ,kernel="rbf",verbose=True)

# params = {'alpha':[0.0001,0.001,0.01,0.1,1,10,50,100]}
# clf = GridSearchCV(Ridge(normalize=True,max_iter=100000), params,  cv = 5,verbose=True)
# X = X[:,1:len(X[0])]
clf.fit(X,y)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
# print clf.best_estimator_
# print clf.best_score_
# print clf.best_params_
# print clf.cv_results_

# print "coef:", clf.coef_
# print clf.n_iter_
# y_pred = clf.predict(X)
# print "y:", y
# print "y_pred:", y_pred
# print (np.sum((y_pred - y)** 2)/len(X))
# print clf.score(X,y)
# print clf.get_params()
joblib.dump(clf, "poly_Rbf_20k.pkl")

# clf = joblib.load('filename.pkl') 
