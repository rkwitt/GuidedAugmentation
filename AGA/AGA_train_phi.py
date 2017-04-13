"""
Train encoder-decoder network (phi) for AGA.
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
    parser.add_option("-v", action="store_true", default=False, dest="verbose",     help="Verbose output.")
    return parser


def train_phi_object_agnostic(config, info, verbose=True):
    """
    Train phi_i^k, i.e., the encoder-decoder network implementing 
    the AGA synthesis function.
    """

    # first, collect data for pre-training
    pretrain_file = os.path.join(config['PATH_TO_MODELS'], 'phi_pre.hdf5')

    pretrain_data = None
    if not os.path.exists(pretrain_file):
        for trn_file in info['phi_train_files']:
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
        info['phi_pre'])
        
    if not os.path.exists(pretrain_model):
        cmd = [config['TORCH'],
            config['TRAIN_PRE'],
            '-dataFile',  pretrain_file,
            '-saveModel', pretrain_model,
            '-batchSize', '256',
            '-epochs', '20',
            '-modelFile', config['phi_def'],
            '-cuda']  
        print cmd
        subprocess.call( cmd )   


    for i, (lo,hi) in enumerate(info['intervals']):

        # get targets for this interval
        targets = info['phi_targets'][i]
        
        models = []
        for target in targets:

            phi_model = "phi_" + \
                os.path.splitext(info['phi_train_files'][i])[0] +\
                "_" + str(target) + ".t7"

            phi_model_log = "phi_" + \
                os.path.splitext(info['phi_train_files'][i])[0] +\
                "_" + str(target) + ".log"

            trn_file = info['phi_train_files'][i]

            cmd = [config['TORCH'],
                config['TRAIN_PHI'],
                '-dataFile',    os.path.join( config['PATH_TO_MODELS'], trn_file ),
                '-modelPHI',    os.path.join( config['PATH_TO_MODELS'], info['phi_pre']),
                '-modelGAMMA',  os.path.join( config['PATH_TO_MODELS'], info['gamma_agnostic_model'] ),
                '-saveModel',   os.path.join( config['PATH_TO_MODELS'], phi_model),
                '-logFile',     os.path.join( config['PATH_TO_MODELS'], phi_model_log),
                '-target',      str( target ), 
                '-cuda',
                '-epochs',      '50',
                '-batchSize',   '64']
            print cmd
            subprocess.call( cmd )

            # add phi_model file to the list of models for this interval
            models.append(phi_model)

        # add models for interval to our list of phi_models
        info['phi_models'].append(tuple(models))

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

    info = train_phi_object_agnostic(config, info)
    
    fid = open(options.output_file,'w')
    yaml.dump(info, fid)
    fid.close()


if __name__ == "__main__":
    main()
