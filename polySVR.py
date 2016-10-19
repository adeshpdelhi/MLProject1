from readFile import readDataSet
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np

data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

# clf = SVR(C=1.0, epsilon=0.2,max_iter=100000000,degree = 3, coef0 = 0 ,kernel="poly",verbose=True)
params = {'kernel':['rbf','linear','poly'],'C':[0.0001,0.001,0.01,0.1,1,10,50,100],'epsilon':[0.2,1,0.001,0.01,10]}
clf = GridSearchCV(SVR(degree=3,max_iter =1000,verbose=True), params,  cv = 5,verbose=True)
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
# joblib.dump(clf, "poly_SVR_20k.pkl")
joblib.dump(clf, "best_SVR_20k.pkl")

# clf = joblib.load('filename.pkl') 
