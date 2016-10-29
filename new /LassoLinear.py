from readFile import readDataSet
from sklearn.linear_model import Lasso
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import numpy as np

data, nrows, ncols = readDataSet("YearPredictionMSD100.txt")
X = data[:,1:91]
y = data[:,0]

X = StandardScaler().fit_transform(X)

# clf = Lasso(alpha = 0.001,max_iter=100000,normalize="True")
params = {'alpha':[1e-6,1e-5,1e-4,0.001,0.01,0.1]}
clf = GridSearchCV(Lasso(max_iter=-1), params,  cv = 5,verbose=True)
clf.fit(X,y)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
print clf.cv_results_['mean_test_score']
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
print clf.cv_results_

data, nrows, ncols = readDataSet("YearPredictionMSDTest10.txt")
X = data[:,1:91]
y = data[:,0]

X = StandardScaler().fit_transform(X)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)

# joblib.dump(clf.best_estimator_, "linear_lasso_20k.pkl")

# clf = joblib.load('filename.pkl') 
