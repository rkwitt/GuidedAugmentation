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

def read_list( file ):
    lines = None
    with open( file ) as fid:
        lines = fid.readlines()
    return [tmp.rstrip() for tmp in lines]


base = sys.argv[1]
objf = sys.argv[2]

data = read_list( objf )
data = data[2:] # ignore __background__ + others

# Train agnostic COR if training/testing files exist
# if os.path.exists( os.path.join( base, 'train.hdf5' ) ) and \
#    os.path.exists( os.path.join( base, 'test.hdf5' ) ):
   
    # cmd = [TORCH,
    #     TRAIN_COR,
    #     '-logFile',         os.path.join( base, 'agnosticCOR.log' ),
    #     '-saveCOR',         os.path.join( base, 'agnosticCOR.t7' ),
    #     '-dataFile',        os.path.join( base, 'train.hdf5' ), 
    #     '-batchSize',       '256',
    #     '-epochs',          '50',
    #     '-column',          '1',
    #     '-cuda']        
    # print cmd
    # subprocess.call(cmd)
        
    # cmd = [TORCH,
    #     TEST_COR,
    #     '-modelCOR',        os.path.join( base, 'agnosticCOR.t7' ),
    #     '-outputFile',      os.path.join( base, 'agnosticCOR_predictions.hdf5' ),
    #     '-dataFile',        os.path.join( base, 'test.hdf5' ),
    #     '-column',          '1',
    #     '-eval',
    #     '-cuda']
    # print cmd
    # subprocess.call(cmd)

    # with h5py.File( os.path.join( base, 'agnosticCOR_predictions.hdf5' ), 'r' ) as hf:

    #     Y_hat = np.asarray( hf.get( 'Y_hat' ) ).reshape(-1)
    #     Y = np.asarray( hf.get( 'Y' ) ).reshape(-1)

    #     print "Agnostic | MSE: %.4f" % ( np.mean( ( Y_hat - Y )**2 ) )

for obj in data:

    path = os.path.join( base, obj )

    if os.path.exists( path + '/train.hdf5' ) and \
       os.path.exists( path + '/test.hdf5' ):

        # cmd = [TORCH,
        #     TRAIN_COR,
        #     '-logFile',         path + '/objectCOR.log',
        #     '-saveCOR',         path + '/objectCOR.t7',
        #     '-dataFile',        path + '/train.hdf5',
        #     '-batchSize',       '64',
        #     '-epochs',          '20',
        #     '-column',          '1', # Depth for now
        #     '-cuda']

        # print cmd
        # subprocess.call(cmd)
        
        # cmd = [TORCH,
        #     TEST_COR,
        #     '-modelCOR',        os.path.join( base, 'agnosticCOR.t7' ),
        #     '-outputFile',      path + '/objectCOR_predictions.hdf5',
        #     '-dataFile',        path + '/test.hdf5',
        #     '-column',          '1',
        #     '-cuda']
        # print cmd
        # subprocess.call(cmd)

        # with h5py.File( path + '/objectCOR_predictions.hdf5', 'r' ) as hf:

        #     Y_hat = np.asarray( hf.get( 'Y_hat' ) ).reshape(-1)
        #     Y = np.asarray( hf.get( 'Y' ) ).reshape(-1)

        #     print "Object: %20s | MSE: %.4f" % ( obj, np.mean( ( Y_hat - Y )**2 ) )

        # Predict with agnostic predictor
        cmd = [TORCH,
            TEST_COR,
            '-modelCOR',        path + '/objectCOR.t7',
            '-outputFile',      path + '/agnosticCOR_predictions.hdf5',
            '-dataFile',        path + '/test.hdf5',
            '-eval',
            '-column',          '1',
            '-cuda']
        print cmd
        subprocess.call(cmd)

        with h5py.File( path + '/agnosticCOR_predictions.hdf5', 'r' ) as hf:

            Y_hat = np.asarray( hf.get( 'Y_hat' ) ).reshape(-1)
            Y = np.asarray( hf.get( 'Y' ) ).reshape(-1)

            print "Agnostic: %20s | MSE: %.4f" % ( obj, np.mean( ( Y_hat - Y )**2 ) )

