"""Learn AE for feature mapping.
"""

# -- generic imports
import numpy as np
import argparse
import logging

# -- keras imports
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives

# -- sklearn imports
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py


def cluster(data, arg_clusters=8, arg_random_state=1234, verbose=False):
	clustering = KMeans(
		n_clusters=arg_clusters,
		init='k-means++',
		random_state=arg_random_state)

	if verbose:
		logging.debug('running k-means with %d clusters [seed=%d]' %
			(arg_clusters, arg_random_state))

	# run k-means
	clustering.fit(data)
	return clustering


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
	i = np.random.randint(0,D.shape[0])
	j = np.random.randint(0,D.shape[0])
	assert (np.abs(D[i,j] - pairwise_distances(X[i,:], Y[j,:])) < 1e-9)

	# allocate space for closest neighbors
	Z = np.zeros(X.shape)

	# set learning target to closest activation in fc7 space
	for i in np.arange(I.shape[0]):
		Z[i,:] = Y[I[i,0],:]
	return Z


def pairing_by_clusters(data, clustering):
	# allocate space
	X = np.zeros(data.shape)

	# get cluster centers
	centers = clustering.cluster_centers_

	# predict cluster center
	idx = clustering.predict(data)

	# set
	for i in np.arange(len(idx)):
		X[i,:] = data[i,:] - centers[idx[i],:]
	return X


# setup parsing
parser = argparse.ArgumentParser(description='Learn AE for feature mapping')
parser.add_argument('--data', dest='data', help='Input data')
parser.add_argument('--output', dest='output', help="Output file")
args = parser.parse_args()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

X_src = None
X_dst = None
with h5py.File(args.data, "r") as f:
	X_src = np.array(f.get("X_trn_src")) # source data
	X_dst = np.array(f.get("X_trn_dst")) # destination data
	X_tst = np.array(f.get("X_tst")) 	 # testing data

# split source data into training/validation
X_src_trn, X_src_val = cross_validation.train_test_split(
		X_src, test_size=0.4, random_state=1234)
p_src_trn, p_src_val = cross_validation.train_test_split(
		np.arange(X_src.shape[0]), test_size=0.4, random_state=1234)

# reduce dimensionality to 500
pca = PCA(n_components=500)
pca.fit(X_src[np.random.choice(X_src.shape[0], 2000, replace=False),:])

X_src = pca.transform(X_src)
X_dst = pca.transform(X_dst)
X_src_trn = pca.transform(X_src_trn)
X_src_val = pca.transform(X_src_val)
X_tst = pca.transform(X_tst)

# pair each observation in X_src_trn/X_src_val with the closest neighbor from X_dst
X_res_trn = pairing_by_neighbor(X_src_trn, X_dst)
X_res_val = pairing_by_neighbor(X_src_val, X_dst)

logging.debug('Source (training) features:   (%d x %d)' % (X_src_trn.shape[0], X_src_trn.shape[1]))
logging.debug('Target (training) features:   (%d x %d)' % (X_res_trn.shape[0], X_res_trn.shape[1]))
logging.debug('Source (validation) features: (%d x %d)' % (X_src_val.shape[0], X_src_val.shape[1]))
logging.debug('Target (validation) features: (%d x %d)' % (X_res_val.shape[0], X_res_val.shape[1]))

# ENCODER/DECODER
dim=X_src_trn.shape[1]
input_data = Input(shape=(dim,))
encoded = Dense(256, activation='linear')(input_data)
encoded = Dense(128, activation='tanh')(encoded)
encoded = Dense( 32, activation='tanh')(encoded)
decoded = Dense(128, activation='tanh')(encoded)
decoded = Dense(256, activation='tanh')(decoded)
decoded = Dense(dim, activation='linear')(decoded)
autoencoder = Model(input=input_data, output=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X_src_trn, X_res_trn, nb_epoch=50, batch_size=512, shuffle=True,
	validation_data=(X_src_val, X_res_val))

# predict fc7 activations of the validation data at the target
X_src_res = autoencoder.predict(X_src_val)

f = h5py.File(args.output, "w")
f.create_dataset("ae_p_src_trn", data=p_src_trn+1)
f.create_dataset("ae_p_src_val", data=p_src_val+1)
f.create_dataset("ae_X_src_val", data=X_src_val)
f.create_dataset("ae_X_src_res", data=X_src_res)
f.create_dataset("X_tst", data=X_tst)
f.close()
