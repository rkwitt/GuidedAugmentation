"""
Train encoder-decoder network (EDN).
"""

import subprocess
import tempfile
import numpy as np
import h5py
import sys
import os

from meta import meta

SUNRGBD_common  = '/scratch2/rkwitt/Mount/images/objects'

TORCH           = '/home/pma/rkwitt/torch/install/bin/th'
TRAIN_PRE       = '/home/pma/rkwitt/GuidedAugmentation/torch/pretrain.lua'
TRAIN_EDN       = '/home/pma/rkwitt/GuidedAugmentation/torch/train_EDN.lua'
TEST_EDN        = '/home/pma/rkwitt/GuidedAugmentation/torch/test_EDN.lua'
MODEL_DEF       = '/home/pma/rkwitt/GuidedAugmentation/torch/models/ae.lua'


def read_file( file_name ):
    """Read line-by-line from file"""
    data = None
    with open( file_name ) as fid:
        data = fid.readlines()

    data = [l.rstrip() for l in data]
    return data


def parse_entry( line ):
    """Parse one line into meta information.

    Format per line:

        obj_name, lo, hi, cv0_file, cv1_file, ..., cvN_file, train_file

    """
    meta_obj = meta()

    parts = line.split( ',' )

    meta_obj.object_name = parts[0]
    meta_obj.covariate_lo, meta_obj.covariate_hi = float( parts[1] ), float( parts[2] )
    [meta_obj.feature_regression_cv_files.append( x ) for x in parts[3:-1]]
    meta_obj.feature_regression_train_file = parts[-1]

    return meta_obj


def set_covariate_targets( meta_objs, range=np.arange(1, 5, 0.5)):
    """Set covariate targets based on covariate data"""

    for meta_obj in meta_objs:
        for r in range:
            if r >= meta_obj.covariate_lo and r <= meta_obj.covariate_hi:
                continue
            meta_obj.covariate_targets.append( r )


def trainer( meta_objs, pretrain=False, do_train=False, do_eval=False ):

    for i, meta_obj in enumerate( meta_objs ):
  
        obj_path = os.path.join( SUNRGBD_common, meta_obj.object_name )

        meta_obj.feature_regression_pretrained_model = os.path.join(obj_path, 'model_pretrain.t7')

        if pretrain and i==0:

            cmd = [TORCH,
                TRAIN_PRE,
                '-dataFile',  os.path.join(obj_path, 'train.hdf5'),
                '-saveModel', meta_obj.feature_regression_pretrained_model,
                '-batchSize', '64',
                '-epochs', '20',
                '-modelFile', MODEL_DEF,
                '-cuda']  
            print cmd
            subprocess.call( cmd )       
    
        # require pretrained model at this point
        found_pretrained_model =  os.path.exists( meta_obj.feature_regression_pretrained_model )
        assert found_pretrained_model == True, 'Pretrained model not found!' 
            
        for t, target in enumerate( meta_obj.covariate_targets ):

            if do_train:

                tmp_name = next(tempfile._get_candidate_names())
                feature_regressor_file = "model_" + tmp_name + '.t7'
                meta_obj.feature_regression_trained_regressors.append( feature_regressor_file )
                
                cmd = [TORCH,
                  TRAIN_EDN,
                  '-dataFile',              os.path.join( obj_path, meta_obj.feature_regression_train_file ),
                  '-modelEDN',              os.path.join( obj_path, 'model_pretrain.t7' ),
                  '-modelCOR',              os.path.join( obj_path, 'modelCOR.t7' ),
                  '-saveModel',             os.path.join( obj_path, feature_regressor_file ),
                  '-target',                str( target ), 
                  '-cuda',
                  '-batchSize',             '64']
                print cmd
                subprocess.call( cmd )
                
            if do_eval:

                for n, cv_feature_file in enumerate(meta_obj.feature_regression_cv_files):

                    tmp_name = next(tempfile._get_candidate_names())
                    tmp_EDNCOR = "model" + tmp_name + '.t7'

                    cmd = [TORCH,
                        TRAIN_EDN,
                        '-dataFile',              os.path.join( obj_path, cv_feature_file + '_train.hdf5' ),
                        '-modelEDN',              os.path.join( obj_path, 'model_pretrain.t7' ),
                        '-modelCOR',              os.path.join( obj_path, 'modelCOR.t7' ),
                        '-saveModel',             os.path.join( '/tmp', tmp_EDNCOR ),
                        '-target',                str( target ), 
                        '-cuda',
                        '-epochs',                '20',
                        '-batchSize',             '64']
                    print cmd
                    subprocess.call( cmd )

                    prediction_file = os.path.splitext( cv_feature_file )[0] + '_test_' + str( target ) + '_prediction.hdf5'

                    cmd = [TORCH,
                        TEST_EDN,
                        '-dataFile',              os.path.join( obj_path, cv_feature_file + '_test.hdf5' ),
                        '-model',                 os.path.join( '/tmp', tmp_EDNCOR ),
                        '-outputFile',            os.path.join( obj_path, prediction_file ) ]
                    print cmd
                    subprocess.call( cmd )
                    
                    with h5py.File( os.path.join( obj_path, prediction_file ),'r') as hf:
                        Y_hat_EDNCOR = np.asarray( hf.get('Y_hat_EDNCOR') )

                        print '--- Feature regression ---'
                        print target, Y_hat_EDNCOR.mean()
                        print meta_obj.covariate_lo, meta_obj.covariate_hi
                        print '--- Feature regression ---'
                    
                    os.remove( os.path.join( '/tmp', tmp_EDNCOR ) )

def main():

    data = {}

    # read file and process line-by-line
    for line in read_file( sys.argv[1] ):

        # parse entry into meta object
        meta_obj = parse_entry( line )

        obj_name = meta_obj.object_name

        if not obj_name in data:
            data[obj_name] = []

        data[obj_name].append( meta_obj )

    for obj_name in data:
        set_covariate_targets( data[obj_name] )

    for obj_name in data:
        trainer( data[obj_name], pretrain=True, do_train=True, do_eval=False )

    # CAUTION: if do_train==False and do_eval==True, uncomment the last lines not
    # to overwrite old file with empty stuff.
    import pickle
    pickle.dump( data, open( sys.argv[2], 'w' ) )


if __name__ == "__main__":
    main()
