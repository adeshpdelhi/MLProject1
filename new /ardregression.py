
from readFile import readDataSet
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn import linear_model

data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

params = {'alpha_1':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100],
		  'alpha_2':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100],
		  'lambda_1':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100],
		  'lambda_2':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100]}
# clf = GridSearchCV(linear_model.BayesianRidge(normalize=True,compute_score = True,verbose=True),params,cv = 5,verbose=True)
clf = linear_model.ARDRegression(normalize=True,compute_score = True,verbose=True,fit_intercept="False")
clf.fit(X,y)
y_pred =  clf.predict(X)
print y
print y_pred
# print clf.best_estimator_
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X)
joblib.dump(clf, "ard.pkl")
# joblib.dump(clf.best_estimator_, "best_SVR_20k.pkl")

# clf = joblib.load('filename.pkl') 
