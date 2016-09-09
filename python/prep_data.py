"""Prepare data for VAE"""

import os
import numpy as np
import argparse
import logging
from sklearn.externals import joblib
from sklearn import cross_validation

# Make things reproducibe
np.random.seed(1234)

# Setup cmdline parsing
parser = argparse.ArgumentParser(description='Data preparation for VAE')
parser.add_argument('--data', dest='data', help='Data file')
parser.add_argument('--meta', dest='meta', help='Meta file')
parser.add_argument('--beg_src', dest='beg_src', type=float, default=0, help='Lower depth value (Source)')
parser.add_argument('--end_src', dest='end_src', type=float, default=1, help='Upper depth value (Source)')
parser.add_argument('--beg_dst', dest='beg_dst', type=float, default=1, help='Lower depth value (Target)')
parser.add_argument('--end_dst', dest='end_dst', type=float, default=2, help='Upper depth value (Target)')
parser.add_argument('--out_dir', dest='out_dir', default='/tmp/', help='Output directory')

args = parser.parse_args()

# Setup cmdline logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

X = np.load(args.data)      # Data
Y = np.loadtxt(args.meta)   # Meta data (scores, depth, etc.)

# Training/Testing split - Everything we do learn here-on will be done on training
X_trn, X_tst = cross_validation.train_test_split(X, test_size=0.4, random_state=1234)
p_trn, p_tst = cross_validation.train_test_split(np.arange(X.shape[0]), test_size=0.4, random_state=1234)

# Split meta data accordingly
Y_trn = Y[p_trn,:]
Y_tst = Y[p_tst,:]

logging.debug('Trn-Data: (%d x %d)' % (X_trn.shape[0], X_trn.shape[1]))
logging.debug('Trn-Meta: (%d x %d)' % (Y_trn.shape[0], Y_trn.shape[1]))
logging.debug('Tst-Data: (%d x %d)' % (X_tst.shape[0], X_tst.shape[1]))
logging.debug('Tst-Meta: (%d x %d)' % (Y_tst.shape[0], Y_tst.shape[1]))

# Write data in MATLAB-readable format to HDD
np.savetxt(os.path.join(args.out_dir, 'X_trn.txt'), X_trn)
np.savetxt(os.path.join(args.out_dir, 'X_tst.txt'), X_tst)
np.savetxt(os.path.join(args.out_dir, 'Y_trn.txt'), Y_trn)
np.savetxt(os.path.join(args.out_dir, 'Y_tst.txt'), Y_tst)

# Depth is the last column
y_trn = Y_trn[:,-1]
y_tst = Y_tst[:,-1]

p_trn_src = np.where((y_trn >= args.beg_src) & (y_trn < args.end_src))[0]
p_trn_dst = np.where((y_trn >= args.beg_dst) & (y_trn < args.end_dst))[0]

# Write indices in MATLAB-readable format to disk
np.savetxt(os.path.join(args.out_dir, 'p_trn.txt'), p_trn+1)
np.savetxt(os.path.join(args.out_dir, 'p_tst.txt'), p_tst+1)
np.savetxt(os.path.join(args.out_dir, 'p_trn_src.txt'), p_trn_src+1)
np.savetxt(os.path.join(args.out_dir, 'p_trn_dst.txt'), p_trn_dst+1)

info = {}
info['beg_src'] = args.beg_src
info['end_src'] = args.beg_src
info['beg_dst'] = args.beg_dst
info['end_dst'] = args.beg_dst
joblib.dump(info,  os.path.join(args.out_dir, 'info.pkl'))

X_trn_src = X_trn[p_trn_src,:]
X_trn_dst = X_trn[p_trn_dst,:]

joblib.dump(X_trn_src, os.path.join(args.out_dir, 'X_trn_src.pkl'))
joblib.dump(X_trn_dst, os.path.join(args.out_dir, 'X_trn_dst.pkl'))

logging.debug('Source features: (%d x %d)' % (X_trn_src.shape[0], X_trn_src.shape[1]))
logging.debug('Target features: (%d x %d)' % (X_trn_dst.shape[0], X_trn_dst.shape[1]))
