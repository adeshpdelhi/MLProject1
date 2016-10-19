from readFile import readDataSet
from sklearn.externals import joblib
from sklearn.linear_model import SGDRegressor 
data, nrows, ncols = readDataSet("YearPredictionMSD20.txt")
X = data[:,1:91]
y = data[:,0]

clf = SGDRegressor(verbose=True, n_iter = 100000,learning_rate="constant", eta0=0.00000001) #eta0=0.00000001
clf.fit(X, y)
print "predict :",clf.predict(X)
print "coef: ", clf.coef_
print clf.score(X, y)
joblib.dump(clf, "SGD_Regressor_20k.pkl")

# clf = joblib.load('filename.pkl') 
