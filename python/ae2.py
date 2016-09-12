"""Learn AE for feature mapping.
"""

# Generic imports
import numpy as np
import argparse
import logging

# Keras imports
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import backend as K
from keras import objectives

# sklearn imports
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import h5py

def cluster(data, n_clusters=8):
	clustering = KMeans(n_clusters=n_clusters, init='k-means++', random_state=1234)
	clustering.fit(data)
	return clustering

def associate(data, clustering):
	X = np.zeros(data.shape)
	Y = clustering.cluster_centers_

	idx = clustering.predict(data)
	for i in np.arange(len(idx)):
		X[i,:] = X[i,:] - Y[idx[i],:] # residual
	return X

# Setup parser
parser = argparse.ArgumentParser(description='Learn AE for feature mapping')
parser.add_argument('--data', dest='data', help='Input data')
args = parser.parse_args()

# Setup logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

X_src, X_dst = None, None
with h5py.File(args.data, "r") as f:
	X_src = np.array(f.get("X_trn_src"))
	X_dst = np.array(f.get("X_trn_dst"))

X_src_trn, X_src_val = cross_validation.train_test_split(
		X_src, 
		test_size=0.4, 
		random_state=1234)
p_src_trn, p_src_val = cross_validation.train_test_split(
		np.arange(X_src.shape[0]), 
		test_size=0.4, 
		random_state=1234)

pca = PCA(n_components=500)
pca.fit(X_src[np.random.choice(X_src.shape[0], 2000, replace=False),:])

X_src = pca.transform(X_src)
X_dst = pca.transform(X_dst)
X_src_trn = pca.transform(X_src_trn)
X_src_val = pca.transform(X_src_val)

clustering = cluster(X_dst, 8)
X_res_trn = associate(X_src_trn, clustering)
X_res_val = associate(X_src_val, clustering)

logging.debug('Source (training) features: (%d x %d)' % (X_src_trn.shape[0], X_src_trn.shape[1]))
logging.debug('Target (training) features: (%d x %d)' % (X_res_trn.shape[0], X_res_trn.shape[1]))
logging.debug('Source (validation) features: (%d x %d)' % (X_src_val.shape[0], X_src_val.shape[1]))
logging.debug('Target (validation) features: (%d x %d)' % (X_res_val.shape[0], X_res_val.shape[1]))

# ENCODER
input_data = Input(shape=(X_src_trn.shape[1],))
encoded = Dense(256, activation='relu')(input_data)
encoded = Dropout(0.5)(encoded)
encoded = Dense(64, activation='tanh')(encoded)

# DECODER
decoded = Dense(256, activation='tanh')(encoded)
decoded = Dropout(0.5)(decoded)
decoded = Dense(X_src_trn.shape[1], activation='linear')(decoded)
autoencoder = Model(input=input_data, output=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X_src_trn, X_res_trn,
                nb_epoch=100,
                batch_size=256,
                shuffle=True,
                validation_data=(X_src_val, X_res_val))

X_src_res = autoencoder.predict(X_src_val)

np.savetxt('/tmp/vae/ae_p_src_trn.txt', p_src_trn+1) # indices for training
np.savetxt('/tmp/vae/ae_p_src_val.txt', p_src_val+1) # indices for validation
np.savetxt('/tmp/vae/ae_X_src_val.txt', X_src_val) # Val. features (src)
np.savetxt('/tmp/vae/ae_X_src_res.txt', X_src_res) # Est. feature residuals (for src->dst)
