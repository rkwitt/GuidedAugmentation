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


# Setup parser
parser = argparse.ArgumentParser(description='Learn AE for feature mapping')
parser.add_argument('--src', dest='src', help='Source features')
parser.add_argument('--dst', dest='dst', help='Target features')
args = parser.parse_args()

# Setup logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

X_src = joblib.load(args.src) # Source features
X_dst = joblib.load(args.dst) # Target features

# Split training data 60/40 into actual training and validation for learning the AE
X_src_trn, X_src_val = cross_validation.train_test_split(X_src, test_size=0.4, random_state=1234)
p_src_trn, p_src_val = cross_validation.train_test_split(np.arange(X_src.shape[0]), test_size=0.4, random_state=1234)

X_rep_avg = X_dst.mean(axis=0) # Representative as average of target features (possibly suboptimal)
X_res_trn = np.tile(X_rep_avg, (X_src_trn.shape[0],1)) - X_src_trn # training residuals
X_res_val = np.tile(X_rep_avg, (X_src_val.shape[0],1)) - X_src_val # validation residuals

logging.debug('Source (training) features: (%d x %d)' % (X_src_trn.shape[0], X_src_trn.shape[1]))
logging.debug('Target (training) features: (%d x %d)' % (X_res_trn.shape[0], X_res_trn.shape[1]))
logging.debug('Source (validation) features: (%d x %d)' % (X_src_val.shape[0], X_src_val.shape[1]))
logging.debug('Target (validation) features: (%d x %d)' % (X_res_val.shape[0], X_res_val.shape[1]))

# ENCODER
input_data = Input(shape=(X_src_trn.shape[1],))
encoded = Dense(512, activation='relu')(input_data)
encoded = Dropout(0.2)(encoded)
encoded = Dense(128, activation='tanh')(encoded)

# DECODER
decoded = Dense(512, activation='tanh')(encoded)
decoded = Dropout(0.2)(decoded)
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
