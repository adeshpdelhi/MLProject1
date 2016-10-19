from readFile import readDataSet
from sklearn.linear_model import Lasso
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np

data, nrows, ncols = readDataSet("YearPredictionMSD.txt")
X = data[:,1:91]
y = data[:,0]

# clf = Lasso(alpha = 0.001,max_iter=100000,normalize="True")
params = {'alpha':[0.0001,0.001,0.01,0.1,1,10,50,100]}
clf = GridSearchCV(Lasso(normalize=True,max_iter=1000000), params,  cv = 5,verbose=True)
clf.fit(X,y)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
print clf.best_estimator_
print clf.best_score_
print clf.best_params_
print clf.cv_results_

# print clf.coef_
# print clf.n_iter_
# y_pred = clf.predict(X)
# print y
# print y_pred
# print (np.sum((y_pred - y)** 2)/len(X))
# print clf.score(X,y)
# print clf.get_params()
joblib.dump(clf, "linear_lasso_full.pkl")

# clf = joblib.load('filename.pkl') 
