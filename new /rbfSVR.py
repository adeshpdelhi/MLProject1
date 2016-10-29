from readFile import readDataSet
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler

data, nrows, ncols = readDataSet("YearPredictionMSDTrain.txt")
X = data[:,1:91]
y = data[:,0]

clf = joblib.load('PCA_Train.pkl') 
X = clf.transform(X)
print "PCA:",X
X = StandardScaler().fit_transform(X)
print "Standarized: ",X
clf = SVR(C=1000.0,gamma=1e-6, epsilon=5,max_iter=-1,kernel="rbf",verbose=True)
# params = {'C':[0.0001,0.001,0.01,0.1,1,10,100]}
# params = {'epsilon':[4,5,10]}
# clf = GridSearchCV(SVR(C=10.0,max_iter=-1,kernel="rbf",verbose=True), params,  cv = 5,verbose=True,n_jobs = 4)
clf.fit(X,y)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
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
joblib.dump(clf, "rbf_SVR_fullTrain.pkl")
