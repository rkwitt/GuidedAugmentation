"""Pair source/target data.
"""

# -- generic imports
import numpy as np
import argparse
import logging

# -- sklearn imports
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py


def pairing_by_neighbor(X, Y):
	"""
	Input: X, Y

	X	shape=(n,d): contains n feature vectors of dimensionality d
	Y 	shape=(m,d): contains m feature vectors of dimensionality d

	Output:

	Z	shape(n,d): contains at the Z[i,:] entry the closest neighbor
					to X[i,:] w.r.t. Y.
	"""
	# first, compute pairwise distances
	D = pairwise_distances(X, Y)

	# next, sort each row
	I = np.argsort(D, axis=1)
	S = np.sort(D, axis=1) # -- DEBUG only

	# make sure the sorting is correct
	min_idx = np.min(D.shape)
	i = np.random.randint(0, min_idx)
	j = np.random.randint(0, min_idx)
	assert (np.abs(D[i,j] - pairwise_distances(X[i,:], Y[j,:])) < 1e-9)

	# allocate space for closest neighbors
	Z = np.zeros(X.shape)

	# set learning target to closest activation in fc7 space
	for i in np.arange(I.shape[0]):
		# Z[i,:] = Y[I[i,0],:] - X[i,:]
		Z[i,:] = Y[I[i,0],:]
	return Z

# setup parsing
parser = argparse.ArgumentParser(description='Data pairing')
parser.add_argument('--dataFile', dest='data_file', help='Data data file')
parser.add_argument('--pairFile', dest='pair_file', help='Pair data file')
args = parser.parse_args()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

tmp = None
X_source = None
X_target = None
with h5py.File(args.data_file, "r") as f:
	X_source_trn = np.array(f.get("X_source_trn")) # source (training) data
	X_source_val = np.array(f.get("X_source_val")) # source (validation) data
	tmp = np.array(f.get("X_target")) # target data

print X_source_trn.shape
print tmp.shape


# pair each activation
X_target = pairing_by_neighbor(X_source_trn, tmp)

f = h5py.File(args.pair_file, "w")
f.create_dataset("X_source_trn", data=X_source_trn)
f.create_dataset("X_source_val", data=X_source_val)
f.create_dataset("X_target", data=X_target)
f.close()
