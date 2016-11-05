"""
Train encoder-decoder network (EDN).
"""

import subprocess
import tempfile
import numpy as np
import yaml
import h5py
import sys
import os


SUNRGBD_common  = '/scratch2/rkwitt/Mount/images/objects'

TORCH           = '/home/pma/rkwitt/torch/install/bin/th'
TRAIN_PRE       = '/home/pma/rkwitt/GuidedAugmentation/torch/pretrain.lua'
TRAIN_EDN       = '/home/pma/rkwitt/GuidedAugmentation/torch/train_EDN.lua'
TEST_EDN        = '/home/pma/rkwitt/GuidedAugmentation/torch/test_EDN.lua'
MODEL_DEF       = '/home/pma/rkwitt/GuidedAugmentation/torch/models/ae.lua'


def train_EDN_object(info):

    for i, obj in enumerate(info):

        obj_path = os.path.join( SUNRGBD_common, obj )
        
        pretrain_data  = os.path.join(obj_path, 'train.hdf5')
        pretrain_model = os.path.join(obj_path, info[obj]['EDN_pre'])

        if not os.path.exists(pretrain_model):
            cmd = [TORCH,
                TRAIN_PRE,
                '-dataFile',  pretrain_data,
                '-saveModel', pretrain_model,
                '-batchSize', '64',
                '-epochs', '20',
                '-modelFile', MODEL_DEF,
                '-cuda']  
            print cmd
            subprocess.call( cmd )   

        intervals = info[obj]['intervals']
        
        for k, interval in enumerate(intervals):

            models = []
            EDN_targets = info[obj]['EDN_targets'][k]

            for target in EDN_targets:

                EDN_model = "EDN_" + \
                    os.path.splitext(info[obj]['EDN_train_files'][k])[0] + \
                    "_" + str(target) + ".t7"
                models.append(EDN_model)

                cmd = [TORCH,
                  TRAIN_EDN,
                  '-dataFile',              os.path.join( obj_path, info[obj]['EDN_train_files'][k] ),
                  '-modelEDN',              os.path.join( obj_path, info[obj]['EDN_pre']),
                  '-modelCOR',              os.path.join( obj_path, info[obj]['COR_object_model'] ),
                  '-saveModel',             os.path.join( obj_path, EDN_model),
                  '-target',                str( target ), 
                  '-cuda',
                  '-epochs',                '20',
                  '-batchSize',             '64']
                print cmd
                subprocess.call( cmd )

            info[obj]['EDN_object_models'].append(tuple(models))
            
    return info


def eval_EDN_object(info):

    # iterate over all object names in info
    for i, obj in enumerate(info):

        obj_path = os.path.join( SUNRGBD_common, obj )
        
        # pretrain
        pretrain_data  = os.path.join(obj_path, 'train.hdf5')
        pretrain_model = os.path.join(obj_path, info[obj]['EDN_pre'])

        if not os.path.exists(pretrain_model):
            cmd = [TORCH,
                TRAIN_PRE,
                '-dataFile',  pretrain_data,
                '-saveModel', pretrain_model,
                '-batchSize', '64',
                '-epochs', '20',
                '-modelFile', MODEL_DEF,
                '-cuda']  
            print cmd
            subprocess.call( cmd )   

        # get all evaluation intervals, i.e., tuples (start,end), ....
        intervals = info[obj]['intervals']
        
        for k, interval in enumerate(intervals):

            # get EDN covariate targets for k-th interval
            EDN_targets = info[obj]['EDN_targets'][k]
            for target in EDN_targets:

                # iterate over all CV folds for k-th interval and current EDN covariate target
                cv_files = info[obj]['EDN_cv_files'][k]        
                for n, cv_file in enumerate(cv_files):

                    tmp_name = next(tempfile._get_candidate_names())
                    EDN_model = "model_" + tmp_name + '.t7'
                    
                    cmd = [TORCH,
                      TRAIN_EDN,
                      '-dataFile',              os.path.join( obj_path, cv_file + '_train.hdf5' ),
                      '-modelEDN',              os.path.join( obj_path, info[obj]['EDN_pre']),
                      '-modelCOR',              os.path.join( obj_path, info[obj]['COR_object_model'] ),
                      '-saveModel',             os.path.join( '/tmp', EDN_model),
                      '-target',                str( target ), 
                      '-cuda',
                      '-epochs',                '20',
                      '-batchSize',             '64']
                    print cmd
                    subprocess.call( cmd )

                    prediction_file = os.path.splitext( cv_file )[0] + '_test_' + str( target ) + '_prediction.hdf5'

                    cmd = [TORCH,
                        TEST_EDN,
                        '-dataFile',              os.path.join( obj_path, cv_file + '_test.hdf5' ),
                        '-model',                 os.path.join( '/tmp', EDN_model ),
                        '-outputFile',            os.path.join( obj_path, prediction_file ) ]
                    print cmd
                    subprocess.call( cmd )
                    
                    with h5py.File( os.path.join( obj_path, prediction_file ),'r') as hf:
                        Y_hat_EDNCOR = np.asarray( hf.get('Y_hat_EDNCOR') )

                        print "{:10s} | {:4f} | {:4f} | [{:2f} {:2f}]".format(
                            obj,
                            target,
                            np.mean(Y_hat_EDNCOR),
                            info[obj]['intervals'][k][0],
                            info[obj]['intervals'][k][1],
                            )
                    os.remove( os.path.join( '/tmp', EDN_model ) )
                    

def train_EDN_agnostic(info):
    pass


def main():

    # load info 
    f = open(sys.argv[1])
    info = yaml.load(f)
    f.close()

    info = train_EDN_object(info)
    
    f = open(sys.argv[2],'w')
    yaml.dump(info, f)
    f.close()

    # DEBUG
    #eval_EDN_object(info)
    
    

if __name__ == "__main__":
    main()
