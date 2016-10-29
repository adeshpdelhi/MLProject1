from readFile import readDataSet
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler

data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

X = StandardScaler().fit_transform(X)
# clf = SVR(C=10.0,gamma=1e-6, epsilon=0.2,max_iter=-1,kernel="rbf",verbose=True)
# params = {'C':[0.0001,0.001,0.01,0.1,1,10,100]}
params = {'gamma':[1e-3,1e-2,1e-1,1e-0,1e1,1e2,1e3,1e4]}
clf = GridSearchCV(SVR(C=10.0, epsilon=0.2,max_iter=-1,kernel="rbf",verbose=True), params,  cv = 5,verbose=True,n_jobs = 3)
clf.fit(X,y)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
print clf.cv_results_

# print clf.support_
# print clf.support_vectors_
# # print clf.coef_
# y_pred = clf.predict(X)
# print y
# print y_pred
# print (np.sum((y_pred - y)** 2)/len(X))
# print clf.score(X,y)
# print clf.get_params()
joblib.dump(clf.best_estimator_, "rbf_SVR_20k.pkl")
