"""Split data"""

import os
import numpy as np
import argparse
import logging
from sklearn.externals import joblib
from sklearn import cross_validation
import h5py

# make things reproducibe
seed = 1234
np.random.seed(seed)

# setup cmdline parsing
parser = argparse.ArgumentParser(description='Data splitting')
parser.add_argument('--dataFile',dest='data_file', help='Data file')
parser.add_argument('--begSrc', dest='beg_src', type=float, default=0, help='Lower depth value (Source)')
parser.add_argument('--endSrc', dest='end_src', type=float, default=1, help='Upper depth value (Source)')
parser.add_argument('--begDst', dest='beg_dst', type=float, default=1, help='Lower depth value (Target)')
parser.add_argument('--endDst', dest='end_dst', type=float, default=2, help='Upper depth value (Target)')
parser.add_argument('--outFile', dest='out_file', default='/tmp/output.hdf5', help='Output file')
args = parser.parse_args()

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

X = None
Y = None
with h5py.File(args.data_file, 'r') as f:
	X = np.array(f.get('data'))
	Y = np.array(f.get('meta'))

# training/Testing split - Everything we do learn here-on will be done on training
X_trn, X_tst = cross_validation.train_test_split(X, test_size=0.4, random_state=seed)
p_trn, p_tst = cross_validation.train_test_split(np.arange(X.shape[0]), test_size=0.4, random_state=seed)

# split meta data accordingly
Y_trn = Y[p_trn,:]
Y_tst = Y[p_tst,:]

logging.debug('Trn-Data: (%d x %d)' % (X_trn.shape[0], X_trn.shape[1]))
logging.debug('Trn-Meta: (%d x %d)' % (Y_trn.shape[0], Y_trn.shape[1]))
logging.debug('Tst-Data: (%d x %d)' % (X_tst.shape[0], X_tst.shape[1]))
logging.debug('Tst-Meta: (%d x %d)' % (Y_tst.shape[0], Y_tst.shape[1]))

# Depth is the last column in the META data
y_trn = Y_trn[:,-1]
y_tst = Y_tst[:,-1]

# get indices for source AND target data wrt depth
p_source = np.where((y_trn >= args.beg_src) & (y_trn < args.end_src))[0]
p_target = np.where((y_trn >= args.beg_dst) & (y_trn < args.end_dst))[0]

X_source = X_trn[p_source,:] # (full) source training data
X_target = X_trn[p_target,:] # (full) target training data

# training/validation split
X_source_trn, X_source_val = cross_validation.train_test_split(X_source, test_size=0.4, random_state=seed)
p_source_trn, p_source_val = cross_validation.train_test_split(np.arange(X_source.shape[0]), test_size=0.4, random_state=seed)

# create HDF5 data file
f = h5py.File(args.out_file, "w")
f.create_dataset("p_trn", data=p_trn+1) # indices of training data
f.create_dataset("p_tst", data=p_tst+1) # indices of testing data
f.create_dataset("p_source", data=p_source+1) # indices of (full) source training data
f.create_dataset("p_target", data=p_target+1) # indices of (full) target training data
f.create_dataset("p_source_trn", data=p_source_val+1) # indices of source training data
f.create_dataset("p_source_val", data=p_source_trn+1) # indices of source validation data

f.create_dataset("X_trn", data=X_trn) # training data
f.create_dataset("X_tst", data=X_tst) # testing data
f.create_dataset("Y_trn", data=Y_trn) # training meta data
f.create_dataset("Y_tst", data=Y_tst) # testing meta data
f.create_dataset("X_source", data=X_source) # (full) source data
f.create_dataset("X_target", data=X_target) # (full) target data
f.create_dataset("X_source_trn", data=X_source_trn) # source training data
f.create_dataset("X_source_val", data=X_source_val) # source validation data
f.close()
