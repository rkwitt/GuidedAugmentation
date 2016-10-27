import os
import sys
import h5py
import subprocess
import numpy as np


TORCH = '/home/pma/rkwitt/torch/install/bin/th'
REGRESSOR = '/home/pma/rkwitt/GuidedAugmentation/torch/regressor.lua'

base = sys.argv[1]
objf = sys.argv[2]

data = None
with h5py.File( objf, 'r' ) as hf:

    data = list( hf.get( 'object_classes' ) )

data = data[2:] # ignore __background__ + others

#data = ['table'];

for obj in data:

    path = os.path.join( base, obj )

    if os.path.exists( path + '/train.hdf5' ) and \
       os.path.exists( path + '/test.hdf5' ):

        cmd = [TORCH,
            REGRESSOR,
            '-logFile',         path + '/log.txt',
            '-saveModel',       path + '/regressor.t7',
            '-evaluationFile',  path + '/evaluation.hdf5',
            '-trainFile',       path + '/train.hdf5',
            '-testFile',        path + '/test.hdf5',
            '-test',
            '-batchSize', '64',
            '-epochs', '20',
            '-cuda',
            '-target', '1',
        ]
        subprocess.call(cmd)

        with h5py.File( path + '/evaluation.hdf5', 'r' ) as hf:

            y_hat = np.asarray( hf.get( 'y_hat' ) ).reshape(-1)
            y_tst = np.asarray( hf.get( 'y_tst' ) ).reshape(-1)

            print "%20s | MSE: %.4f" % ( obj, np.mean( (y_hat-y_tst)**2 ) )
