"""Object feature augmentation."""

# generic imports
import scipy.io as sio
import subprocess
from optparse import OptionParser
import numpy as np
import pickle
import h5py
import sys
import io
import os

# import class information for meta class
from meta import meta

TORCH = '/home/pma/rkwitt/torch/install/bin/th'
TEST_COR = '/home/pma/rkwitt/GuidedAugmentation/torch/test_COR.lua'
TEST_EDN = '/home/pma/rkwitt/GuidedAugmentation/torch/test_EDN.lua'


def setup_parser():
    """Setup the CLI parsing."""
    parser = OptionParser()
    parser.add_option("-l", "--list", help="File with image file names.")
    parser.add_option("-m", "--meta", help="Pickled META information file.")
    parser.add_option("-b", "--base", help="Base directory of SUNRGBD images.")
    parser.add_option("-c", "--objc", help="File with object classes.")
    parser.add_option("-t", "--temp", help="Temporary directory")

    parser.add_option("-v", action="store_true", default=False, dest="verbose", help="Verbose output.")
    return parser


def read_list( file ):
    lines = None
    with open( file ) as fid:
        lines = fid.readlines()
    return [tmp.rstrip() for tmp in lines]


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = setup_parser()
    (options, args) = parser.parse_args()

    # load meta information
    info = pickle.load( open( options.meta, 'r' ) )

    if options.verbose:
        print "Available objects:"
        for obj_name in info:
            print obj_name

    image_files = read_list( options.list )

    object_classes = read_list( options.objc )

    if options.verbose:
        print "Processing %d images" % len( image_files )

    for img_file in image_files:

        EDN_X = None # EDN synthesized features
        EDN_0 = np.zeros( (1,4+4096) ) # dummy row of all zeros

        EDN_X_file = os.path.join( options.base, os.path.splitext(img_file)[0] + '_EDN.mat')

        img_file_no_ext = os.path.splitext(img_file)[0]

        if options.verbose:
            print img_file

        # Read HDF5 file + get object name(s)
        img_data_file = img_file_no_ext + '_data.hdf5'
            
        if not os.path.exists( os.path.join( options.base, img_data_file ) ):
            print "skipping %s" % img_data_file
            continue

        # Read object IDs for all features in the data file
        object_ids = None
        X = None
        Y = None
        with h5py.File( os.path.join( options.base, img_data_file) ,'r') as hf:
            object_ids = np.asarray( hf.get('objects'), dtype=int)
            X = np.asarray( hf.get( 'X' ) )
            Y = np.asarray( hf.get( 'Y' ) )
    
        # iterate over the detected objects in the image
        for cnt, object_class_ID in enumerate( object_ids ):

            object_name = object_classes[object_class_ID-1]

            """
            Strategy:

                1. Gather the correct covariate regressor (COR)
                2. Use COR to predict covariate value for all features
                3. Select the corresponding EDNs 
                4. Call EDNs for all available covariate targets
                5. Assemble synthesized features and write to file
            """
            if object_name in info:
                
                model_COR = os.path.join( options.base, 
                    'objects', 
                    object_name, 
                    'modelCOR.t7')
                assert os.path.exists( model_COR )
                
                tmp_data_file = os.path.join( options.temp, 'X.hdf5')
                tmp_pred_file = os.path.join( options.temp, 'prediction.hdf5')

                with h5py.File( tmp_data_file, 'w') as hf:
                    if len(X.shape) == 1: # only one entry 
                        X = X.reshape((X.shape[0],1))
                        Y = Y.reshape((Y.shape[0],1))

                    hf.create_dataset('/X', data=X[:,cnt]) # MATLAB writes matrix in transposed format
                    hf.create_dataset('/Y', data=Y[:,cnt]) # MATLAB writes matrix in transposed format

                cmd = [TORCH,
                    TEST_COR,
                    '-modelCOR',        model_COR,
                    '-dataFile',        tmp_data_file,
                    '-outputFile',      tmp_pred_file,
                    '-column',          '1'] # Depth for now
                print cmd
                subprocess.call( cmd )

                # read prediction
                Y_hat = None
                with h5py.File( tmp_pred_file ,'r') as hf:
                    Y_hat = np.asarray( hf.get('Y_hat')).reshape(-1)

                for tmp in info[object_name]:
                    lo = tmp.covariate_lo
                    hi = tmp.covariate_hi

                    if (Y_hat >= lo and Y_hat <= hi):
                        
                        for EDN_i in np.arange( len( tmp.feature_regression_trained_regressors ) ):

                            tmp_out_file = os.path.join( options.temp, 'tmp_out_file.hdf5')

                            model_EDNCOR = os.path.join(
                                options.base, 
                                'objects',
                                object_name,
                                tmp.feature_regression_trained_regressors[EDN_i])

                            cmd = [TORCH,
                                TEST_EDN,
                                '-dataFile',    tmp_data_file,
                                '-model',       model_EDNCOR,
                                '-outputFile',  tmp_out_file]
                            print cmd
                            subprocess.call( cmd )

                            """
                                1. Read EDN output
                                2. Store EDN output X_hat as new row in END_X [object_class_ID, Y_hat, X]
                            """
                            with h5py.File( tmp_out_file ,'r') as hf:
                                X_hat = np.asarray( hf.get('X_hat') )

                                EDN_0[0,0]  = object_class_ID               # Object class (starting at 1)
                                EDN_0[0,1]  = Y_hat                         # EDNCOR output
                                EDN_0[0,2]  = tmp.covariate_targets[EDN_i]  # Covariate target
                                EDN_0[0,3]  = cnt+1                         # Object counter (starting at 1)
                                EDN_0[0,4:] = X_hat                         # EDN output
                                if EDN_X is None:
                                    EDN_X = EDN_0
                                else:
                                    EDN_X = np.vstack((EDN_X, EDN_0))

        if not EDN_X is None:
            sio.savemat(EDN_X_file, {'X':EDN_X})
        
        
if __name__ == '__main__':
    sys.exit( main() )
