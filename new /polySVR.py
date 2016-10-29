from readFile import readDataSet
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler

data, nrows, ncols = readDataSet("YearPredictionMSD100.txt")
X = data[:,1:91]
y = data[:,0]

X = StandardScaler().fit_transform(X)
clf = SVR(C=10.0, epsilon=0.2,max_iter=-1,degree = 3, coef0 = 0 ,kernel="linear",verbose=True)
# params = {'kernel':['rbf','linear','poly'],'C':[0.0001,0.001,0.01,0.1,1,10,50,100],'epsilon':[0.2,1,0.001,0.01,10]}
# params = {'C':[0.0001,0.001,1,10,100]}
# clf = GridSearchCV(SVR(degree=3,max_iter =10000,verbose=True), params,  cv = 5,verbose=True, n_jobs = 3)
clf.fit(X,y)
# y_pred = clf.predict(X)
# print y_pred, y
# print (np.sum((y_pred - y)** 2)/len(X))
# print clf.score(X,y)
# print clf.best_estimator_
# print clf.best_score_
# print clf.best_params_
# print clf.cv_results_

print clf.support_
print clf.support_vectors_
# print clf.coef_
y_pred = clf.predict(X)
print y
print y_pred
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
print clf.get_params()

data, nrows, ncols = readDataSet("YearPredictionMSDTest10.txt")
X = data[:,1:91]
y = data[:,0]

X = StandardScaler().fit_transform(X)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)

# joblib.dump(clf, "poly_SVR_20k.pkl")
# joblib.dump(clf.best_estimator_, "best_SVR_20k.pkl")

# clf = joblib.load('filename.pkl') 
