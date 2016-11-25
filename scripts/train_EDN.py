"""
Train encoder-decoder network (EDN).
"""

from optparse import OptionParser
import subprocess
import tempfile
import numpy as np
import yaml
import h5py
import sys
import os

from utils import read_config


def setup_parser():
    """
    Setup the CLI parsing.
    """
    parser = OptionParser()
    parser.add_option("-c", "--config_file",                                        help="YAML config file.")
    parser.add_option("-y", "--info_file",                                          help="YAML information file.")
    parser.add_option("-o", "--output_file",                                        help="YAML output file with model information.")
    parser.add_option("-a", action="store_true", default=False, dest="agnostic",    help="Switch to agnostic")
    parser.add_option("-v", action="store_true", default=False, dest="verbose",     help="Verbose output.")
    return parser


def train_EDN_object_agnostic(config, info, verbose=True):

    # First, collect data for pre-training
    pretrain_file = os.path.join( 
        config['PATH_TO_MODELS'], 
        'EDN_pre.hdf5')

    pretrain_data = None
    if not os.path.exists(pretrain_file):

        for trn_file in info['EDN_train_files']:
            with h5py.File( os.path.join( config['PATH_TO_MODELS'], trn_file ), 'r') as hf:
                X = np.asarray( hf.get('X') ).transpose()
                if verbose:
                    print "Adding {} x {} data from {}".format(
                        X.shape[0],
                        X.shape[1],
                        trn_file
                        )
                if pretrain_data is None:
                    pretrain_data = X
                else:
                    pretrain_data = np.vstack((pretrain_data, X))

        with h5py.File( pretrain_file, 'w') as hf:
            hf.create_dataset('/X', data=pretrain_data.transpose())

    
    pretrain_model = os.path.join(
        config['PATH_TO_MODELS'],
        info['EDN_pre'])
        
    if not os.path.exists(pretrain_model):
        cmd = [config['TORCH'],
            config['TRAIN_PRE'],
            '-dataFile',  pretrain_file,
            '-saveModel', pretrain_model,
            '-batchSize', '256',
            '-epochs', '20',
            '-modelFile', config['EDN_def'],
            '-cuda']  
        print cmd
        subprocess.call( cmd )   


    for i, (lo,hi) in enumerate(info['intervals']):

        # get targets for this interval
        targets = info['EDN_targets'][i]
        
        models = []
        for target in targets:

            EDN_model = "EDN_" + \
                os.path.splitext(info['EDN_train_files'][i])[0] +\
                "_" + str(target) + ".t7"

            trn_file = info['EDN_train_files'][i]

            cmd = [config['TORCH'],
                config['TRAIN_EDN'],
                '-dataFile',    os.path.join( config['PATH_TO_MODELS'], trn_file ),
                '-modelEDN',    os.path.join( config['PATH_TO_MODELS'], info['EDN_pre']),
                '-modelAR',     os.path.join( config['PATH_TO_MODELS'], info['AR_agnostic_model'] ),
                '-saveModel',   os.path.join( config['PATH_TO_MODELS'], EDN_model),
                '-target',      str( target ), 
                '-cuda',
                '-epochs',      '50',
                '-batchSize',   '64']
            print cmd
            subprocess.call( cmd )

            # add EDN_model file to the list of models for this interval
            models.append(EDN_model)

        # add models for interval to our list of EDN_models
        info['EDN_models'].append(tuple(models))

    return info


def train_EDN_object(config, info):

    for i, obj in enumerate(info):

        obj_path = os.path.join( config['PATH_TO_MODELS'], obj )
        
        pretrain_data  = os.path.join( obj_path, 'train.hdf5' )
        pretrain_model = os.path.join( obj_path, info[obj]['EDN_pre'] )

        if not os.path.exists(pretrain_model):
            cmd = [config['TORCH'],
                config['TRAIN_PRE'],
                '-dataFile',  pretrain_data,
                '-saveModel', pretrain_model,
                '-batchSize', '64',
                '-epochs', '20',
                '-modelFile', config['EDN_def'],
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

                cmd = [config['TORCH'],
                  config['TRAIN_EDN'],
                  '-dataFile',              os.path.join( obj_path, info[obj]['EDN_train_files'][k] ),
                  '-modelEDN',              os.path.join( obj_path, info[obj]['EDN_pre']),
                  '-modelAR',              os.path.join( obj_path, info[obj]['AR_object_model'] ),
                  '-saveModel',             os.path.join( obj_path, EDN_model),
                  '-target',                str( target ), 
                  '-cuda',
                  '-epochs',                '20',
                  '-batchSize',             '64']
                print cmd
                subprocess.call( cmd )

            info[obj]['EDN_object_models'].append(tuple(models))
            
    return info
              

def main():

    parser = setup_parser()
    (options, args) = parser.parse_args()

    config = read_config(
       options.config_file, 
       verbose=options.verbose)

    fid = open(options.info_file)
    info = yaml.load(fid)
    fid.close()

    # train in object agnostic manner
    if options.agnostic:
        info = train_EDN_object_agnostic(config, info)
    # train in object specific manner
    else:
        info = train_EDN_object(config, info)
    
    f = open(options.output_file,'w')
    yaml.dump(info, f)
    f.close()


if __name__ == "__main__":
    main()
