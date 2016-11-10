"""
GOA - Guided Object Augmentation

rkwitt, mdixit, 2016
"""


import os
import sys
import uuid 
import yaml
import h5py
import scipy.io as sio
import subprocess
import argparse
import numpy as np


def setup_parser():
    """
    Setup the CLI parsing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_config",         metavar='', help="YAML system configuration file")
    parser.add_argument("--edn_config",         metavar='', help="YAML EDN/COR configuration file")    
    parser.add_argument("--input_file",         metavar='', help="MATLAB input file with features/scores")
    parser.add_argument("--label_file",         metavar='', default=None, help="MATLAB file with object labels of features")
    parser.add_argument("--output_file",        metavar='', help="MATLAB output file with synthetic features")   
    parser.add_argument("--verbose",            action="store_true", default=False, dest="verbose",     help="Verbose output.")
    return parser


def read_file(file, verbose=False):
    """
    Read file line by line into list.

    Input: filename

    Output: list of whitespace-stripped lines
    """
    lines = None
    with open( file ) as fid:
        lines = fid.readlines()
    if verbose:
        print "Read {} with {} entries.".format(file, len(lines))
    return [tmp.rstrip() for tmp in lines]


def read_config(file, verbose=False):
    """
    Read system config YAML file.

    Input: YAML config file

    Returns: dict
    """
    if verbose:
        print "Read config file {}".format(file)
    fid = open(file, "r")
    config = yaml.load(fid)
    fid.close()
    return config


def read_RCNN_mat_file(file, verbose=False):
    """
    Read features from MATLAB file.

    Input: MATLAB .mat file with fields 
        -'CNN_features' 
        -'CNN_scores'

    Returns: (features, scores)
    """
    mat = sio.loadmat(file)
    if verbose:
        print 'Read {}x{} CNN features and {}x{} scores'.format(
            mat['CNN_feature'].shape[0],
            mat['CNN_feature'].shape[1],
            mat['CNN_scores'].shape[0],
            mat['CNN_scores'].shape[1])

    return mat['CNN_feature'], mat['CNN_scores']


def implement_object_agnostic_covariate_estimate(config, X, model):

    tmp_data_file = os.path.join( config['TEMP_DIR'], 'COR_X_'+str(uuid.uuid4())+'.hdf5')
    tmp_pred_file = os.path.join( config['TEMP_DIR'], 'COR_X_'+str(uuid.uuid4())+'_prediction.hdf5')

    with h5py.File( tmp_data_file, 'w') as hf:
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0],1))
        hf.create_dataset('/X', data=X) 
    
    cmd = [
        config['TORCH'],
        config['TEST_COR'],
        '-modelCOR', model,
        '-dataFile', tmp_data_file,
        '-outputFile', tmp_pred_file]
    subprocess.call( cmd )

    vals = None
    with h5py.File( tmp_pred_file ,'r') as hf:
        vals = np.asarray( hf.get('Y_hat')).reshape(-1)

    os.remove(tmp_data_file)
    os.remove(tmp_pred_file)

    return vals


def object_agnostic_covariate_estimate(config, info, object_features, object_labels, verbose=False):
    vals = []
    if object_features is None:
        return vals

    N = object_features.shape[0]
    
    # iterate over all features
    for i in np.arange(N):
        
        # get object agnostic COR model
        model_COR = os.path.join(
            config['PATH_TO_MODELS'], 
            'agnosticCOR.t7')

        # call actual implementation
        tmp_val = implement_object_agnostic_covariate_estimate(
            config, 
            object_features[i,:], 
            model_COR)
        [vals.append(x) for x in tmp_val]
    return vals


def implement_object_synthesize(config, X, model):
    tmp_data_file = os.path.join( config['TEMP_DIR'], 'EDN_X_'+str(uuid.uuid4())+'.hdf5')
    tmp_pred_file = os.path.join( config['TEMP_DIR'], 'EDN_Xhat_'+str(uuid.uuid4())+'.hdf5')
    
    with h5py.File( tmp_data_file, 'w') as hf:
        if len(X.shape) == 1: # only one entry 
            X = X.reshape((X.shape[0],1))
        hf.create_dataset('/X', data=X) 

    cmd = [
        config['TORCH'],
        config['TEST_EDN'],
        '-dataFile', tmp_data_file,
        '-model', model,
        '-outputFile', tmp_pred_file]
    subprocess.call( cmd )
   
    X_hat = None # Synthesized RCNN feature
    Y_hat = None # Covariate as predicted for the synthesized feature
    with h5py.File( tmp_pred_file ,'r') as hf:
        X_hat = np.asarray( hf.get('X_hat'))
        Y_hat = np.asarray( hf.get('Y_hat_EDNCOR'))

    os.remove(tmp_data_file)
    os.remove(tmp_pred_file)

    return X_hat, Y_hat


def object_agnostic_synthesize(config, info, object_features, object_labels, covariate_estimates, verbose=False):
    EDN_X = None                # synthesized features
    EDN_0 = np.zeros((1,4096))  # dummy
    SUP_X = None                # synthesized suppl. information
    SUP_0 = np.zeros((1,4))     # dummy

    N = object_features.shape[0]
    
    for i in np.arange(N):
        for j, (lo,hi) in enumerate(info['intervals']):
            if (covariate_estimates[i] >= lo and covariate_estimates[i] <= hi):
                for t, target in enumerate(info['EDN_targets'][j]):
                    model_EDNCOR = os.path.join(
                        config['PATH_TO_MODELS'], 
                        info['EDN_models'][j][t])

                    tmp_X, tmp_Y = implement_object_synthesize(
                        config, 
                        object_features[i,:], 
                        model_EDNCOR)

                    SUP_0[0,0]  = object_labels[i]
                    SUP_0[0,1]  = tmp_Y
                    SUP_0[0,2]  = target
                    SUP_0[0,3]  = i
                    
                    if EDN_X is None:
                        EDN_X = tmp_X
                        SUP_X = SUP_0
                    else:
                        EDN_X = np.vstack((EDN_X, tmp_X))
                        SUP_X = np.vstack((SUP_X, SUP_0))
    
        # catch the case when estimated covariate > highest training covariate
        if covariate_estimates[i] > hi:
            print "Covariate {} > HI".format(covariate_estimates[i])
            for t, target in enumerate(info['EDN_targets'][-1]):
                model_EDNCOR = os.path.join(
                    config['PATH_TO_MODELS'], 
                    info['EDN_models'][-1][t])
                tmp_X, tmp_Y = implement_object_synthesize(
                        config, 
                        object_features[i,:], 
                        model_EDNCOR)

                SUP_0[0,0]  = object_labels[i]
                SUP_0[0,1]  = tmp_Y
                SUP_0[0,2]  = target
                SUP_0[0,3]  = i
                    
                if EDN_X is None:
                    EDN_X = tmp_X
                    SUP_X = SUP_0
                else:
                    EDN_X = np.vstack((EDN_X, tmp_X))
                    SUP_X = np.vstack((SUP_X, SUP_0))  


    if verbose and not EDN_X is None:
        print 'Synthesized {} x {} RCNN features'.format(
            EDN_X.shape[0], EDN_X.shape[1])

    return EDN_X, SUP_X


def main(argv=None):
    if argv is None:
        argv = sys.argv

    options = setup_parser().parse_args()

    sys_config = read_config(options.sys_config, verbose=options.verbose)
    edn_config = read_config(options.edn_config, verbose=options.verbose)

    object_features, object_scores = read_RCNN_mat_file(options.input_file, options.verbose)
 
    if object_features is None:
        print "No features extracted!"
        sys.exit(-1)

    object_labels = sio.loadmat(options.label_file)['labels'].reshape(-1)
    object_labels = [label for label in object_labels]

    assert len(object_labels) == object_features.shape[0], 'Dimensionality mismatch!'

    covariate_estimates = object_agnostic_covariate_estimate(
        sys_config,
        edn_config,
        object_features,
        object_labels,
        options.verbose) 

    synthetic_features, metadata = object_agnostic_synthesize(
        sys_config,
        edn_config,
        object_features,
        object_labels,
        covariate_estimates,
        options.verbose)
 
    if not synthetic_features is None:
        sio.savemat(options.output_file, 
            {
                'CNN_feature': synthetic_features, 
                'CNN_metadata' : metadata
            })

if __name__ == '__main__':
    sys.exit( main() )