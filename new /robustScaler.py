from sklearn.preprocessing import RobustScaler
from readFile import readDataSet
from sklearn.linear_model import Ridge
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np


data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

X = RobustScaler().fit_transform(X)



clf = Ridge(alpha = 0.001,max_iter=-1,normalize="True")
# params = {'alpha':[0.0001,0.001,0.01,0.1,1,10,50,100]}
# clf = GridSearchCV(Ridge(normalize=True,max_iter=100000), params,  cv = 5,verbose=True)
clf.fit(X,y)
y_pred = clf.predict(X)
print y_pred, y
print (np.sum((y_pred - y)** 2)/len(X))
print clf.score(X,y)
# print clf.best_estimator_
# print clf.best_score_
# print clf.best_params_
# print clf.cv_results_

# print clf.coef_
# print clf.n_iter_
# y_pred = clf.predict(X)
# print y
# print y_pred
# print (np.sum((y_pred - y)** 2)/len(X))
# print clf.score(X,y)
# print clf.get_params()
# f = open("y.txt","w")
# f.write("2")
# f.close()
# joblib.dump(clf.best_estimator_, "linear_Ridge_full.pkl")

# clf = joblib.load('filename.pkl') 

