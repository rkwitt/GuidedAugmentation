"""
Evaluate SVR regression for depth based on Fast-RCNN features extracted
from GT bounding boxes.
"""

import os
import sys
import pickle
import numpy as np
from optparse import OptionParser
import ConfigParser

# sklearn imports
from sklearn.svm import SVR
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn import grid_search

# indices start at 1 when exported in MATLAB
pos = np.loadtxt(sys.argv[1])-1
pos = list(pos)

# load features
X = np.load(sys.argv[2])
X = X[pos,:]

# load regressor variable
y = np.loadtxt(sys.argv[3])

# Nr. of CV runs
n_splits = 10

r2s = np.zeros((n_splits,1)) # R2 score
mse = np.zeros((n_splits,1)) # Mean squared error
mae = np.zeros((n_splits,1)) # Mean absolute error

# run n_splits CV runs
for i in xrange(n_splits):

    # create train/test split
    X_trn, X_tst, y_trn, y_tst = cross_validation.train_test_split(
        X, y, test_size=0.4, random_state=i)

    # DEBUG: reduce dimensions via PCA
    pca = PCA(n_components=128)
    pca.fit(X_trn)

    # DEBUG: transform samples to low-dim. space
    X_trn_rdim = pca.transform(X_trn)
    X_tst_rdim = pca.transform(X_tst)

    # X_trn_rdim = X_trn
    # X_tst_rdim = X_tst

    # train (linear) support vector regressor
    svr = SVR(C=1.0, kernel='rbf')
    svr.fit(X_trn_rdim, y_trn)

    # predict depth
    y_pred = svr.predict(X_tst_rdim)

    dbg = np.zeros((len(y_pred),2))
    dbg[:,0] = y_tst
    dbg[:,1] = y_pred
    dbg_file = os.path.join('/tmp/', 'iter_%.4d.txt' % i)
    np.savetxt(dbg_file, dbg)


    # compute performance measures
    r2s[i] =            r2_score(y_pred, y_tst)
    mse[i] =  mean_squared_error(y_pred, y_tst)
    mae[i] = mean_absolute_error(y_pred, y_tst)

    print r2s[i], mse[i], mae[i]

print "R2  score:", np.mean(r2s)
print "MSE score:", np.mean(mse)
print "MAE score:", np.mean(mae)
