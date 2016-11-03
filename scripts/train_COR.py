"""Train covariate regressor (COR)
"""

import os
import sys
import h5py
import subprocess
import numpy as np


TORCH     = '/home/pma/rkwitt/torch/install/bin/th'
TRAIN_COR = '/home/pma/rkwitt/GuidedAugmentation/torch/train_COR.lua'
TEST_COR  = '/home/pma/rkwitt/GuidedAugmentation/torch/test_COR.lua'


base = sys.argv[1]
objf = sys.argv[2]

data = None
with h5py.File( objf, 'r' ) as hf:

    data = list( hf.get( 'object_classes' ) )

data = data[2:] # ignore __background__ + others

for obj in data:

    path = os.path.join( base, obj )

    if os.path.exists( path + '/train.hdf5' ) and \
       os.path.exists( path + '/test.hdf5' ):

        cmd = [TORCH,
            TRAIN_COR,
            '-logFile',         path + '/train_COR.log',
            '-saveCOR',         path + '/modelCOR.t7',
            '-dataFile',        path + '/train.hdf5',
            '-batchSize',       '64',
            '-epochs',          '20',
            '-column',          '1', # Depth for now
            '-cuda']

        print cmd
        subprocess.call(cmd)
        
        cmd = [TORCH,
            TEST_COR,
            '-modelCOR',        path + '/modelCOR.t7',
            '-outputFile',      path + '/predictions.hdf5',
            '-dataFile',        path + '/test.hdf5',
            '-column',          '1', # Depth for now
            '-cuda']
        print cmd
        subprocess.call(cmd)

        with h5py.File( path + '/predictions.hdf5', 'r' ) as hf:

            Y_hat = np.asarray( hf.get( 'Y_hat' ) ).reshape(-1)
            Y = np.asarray( hf.get( 'Y' ) ).reshape(-1)

            print "%20s | MSE: %.4f" % ( obj, np.mean( ( Y_hat - Y )**2 ) )
