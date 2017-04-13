"""
AGA - Attribute-Guided Augmentation

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

from utils import read_file, read_config


def setup_parser():
    """
    Setup the CLI parsing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sys_config",         metavar='', help="YAML system configuration file")
    parser.add_argument("--aga_config",         metavar='', help="YAML AGA configuration file")    
    parser.add_argument("--input_file",         metavar='', help="MATLAB input file with features/scores")
    parser.add_argument("--label_file",         metavar='', default=None, help="MATLAB file with object labels of all features")
    parser.add_argument("--output_file",        metavar='', help="MATLAB output file with synthetic features")   
    parser.add_argument("--verbose",            action="store_true", default=False, dest="verbose",     help="Verbose output.")
    return parser


def read_RCNN_mat_file(file, verbose=False):
    """
    Read features from MATLAB file.

    Input: MATLAB .mat file with fields 
        -'CNN_features' 
        [-'CNN_scores'] - optional

    Returns: features
    """
    mat = sio.loadmat(file)
    
    if verbose:
        print 'Read {}x{} CNN features'.format(
            mat['CNN_feature'].shape[0],
            mat['CNN_feature'].shape[1])

    return mat['CNN_feature'], None


def implement_object_agnostic_covariate_estimate(config, X, model):
    """
    Attribute prediction IMPLEMENTATION
    """

    # temporary files that are created during attribute prediction
    tmp_data_file = os.path.join( config['TEMP_DIR'], 'GAMMA_X_'+str(uuid.uuid4())+'.hdf5')
    tmp_pred_file = os.path.join( config['TEMP_DIR'], 'GAMMA_X_'+str(uuid.uuid4())+'_prediction.hdf5')

    # create suitable input file
    with h5py.File( tmp_data_file, 'w') as hf:
        if len(X.shape) == 1:
            X = X.reshape((X.shape[0],1))
        hf.create_dataset('/X', data=X) 
    
    # call predictor
    cmd = [
        config['TORCH'],
        config['TEST_GAMMA'],
        '-model', model,
        '-dataFile', tmp_data_file,
        '-outputFile', tmp_pred_file]
    subprocess.call( cmd )

    # read predictions (one prediction per feature x)
    vals = None
    with h5py.File( tmp_pred_file ,'r') as hf:
        vals = np.asarray( hf.get('Y_hat')).reshape(-1)

    # cleanup tmp files
    os.remove(tmp_data_file)
    os.remove(tmp_pred_file)

    # return predictions
    return vals


def object_agnostic_covariate_estimate(config, info, object_features, object_labels, verbose=False):
    """
    Attribute prediction LOGIC
    """
    vals = []
    if object_features is None:
        return vals

    N = object_features.shape[0]
    
    # iterate over all features
    for i in np.arange(N):
        
        # get object agnostic COR model
        model_AR = os.path.join(
            config['PATH_TO_MODELS'], 
            'agnosticAR.t7')

        # call actual implementation
        tmp_val = implement_object_agnostic_covariate_estimate(
            config, 
            object_features[i,:], 
            model_AR)
        [vals.append(x) for x in tmp_val]

    return vals


def implement_object_agnostic_synthesize(config, X, model):
    """
    Feature synthesis IMPLEMENTATION
    """

    # tmp. files created during synthesis
    tmp_data_file = os.path.join( config['TEMP_DIR'], 'PHI_X_'+str(uuid.uuid4())+'.hdf5')
    tmp_pred_file = os.path.join( config['TEMP_DIR'], 'PHI_Xhat_'+str(uuid.uuid4())+'.hdf5')
    
    # create suitable input file
    with h5py.File( tmp_data_file, 'w') as hf:
        if len(X.shape) == 1: # only one entry 
            X = X.reshape((X.shape[0],1))
        hf.create_dataset('/X', data=X) 

    # exec. synthesis 
    cmd = [
        config['TORCH'],
        config['TEST_PHI'],
        '-dataFile', tmp_data_file,
        '-model', model,
        '-outputFile', tmp_pred_file]
    subprocess.call( cmd )
   
    X_hat = None # Synthesized RCNN feature
    Y_hat = None # Covariate as predicted for the synthesized feature
    with h5py.File( tmp_pred_file ,'r') as hf:
        X_hat = np.asarray( hf.get('X_hat'))
        Y_hat = np.asarray( hf.get('Y_hat_PhiGamma'))

    # remove tmp. files
    os.remove(tmp_data_file)
    os.remove(tmp_pred_file)

    # return AGA-syn. features + values of gamma for AGA-syn. features
    return X_hat, Y_hat


def object_agnostic_synthesize(config, info, object_features, object_labels, covariate_estimates, verbose=False):
    """
    Feature synthesis LOGIC
    """
    PHI_X = None                # synthesized features
    PHI_0 = np.zeros((1,4096))  # dummy
    SUP_X = None                # synthesized suppl. information
    SUP_0 = np.zeros((1,4))     # dummy

    N = object_features.shape[0]
    
    # Iterate over all features
    for i in np.arange(N):

        # Iterate over all intervals [l_i,h_i]
        for j, (lo,hi) in enumerate(info['intervals']):

            # Check if \hat{t} is in [l_i,h_i]
            if (covariate_estimates[i] >= lo and covariate_estimates[i] <= hi):
                
                # Call \phi_i^t for all possible attribute target values
                for t, target in enumerate(info['PHI_targets'][j]):
                    model_PhiGamma = os.path.join(
                        config['PATH_TO_MODELS'], 
                        info['PHI_models'][j][t])

                    tmp_X, tmp_Y = implement_object_agnostic_synthesize(
                        config, 
                        object_features[i,:], 
                        model_PhiGamma)

                    SUP_0[0,0]  = object_labels[i]  # label for object feature
                    SUP_0[0,1]  = tmp_Y             # gamma(.) for AGA-syn. feature
                    SUP_0[0,2]  = target            # desired attribute value
                    SUP_0[0,3]  = i                 # ID of currently synthesized feature

                    # stack AGA-synthesized features together
                    if PHI_X is None:
                        PHI_X = tmp_X
                        SUP_X = SUP_0
                    else:
                        PHI_X = np.vstack((PHI_X, tmp_X))
                        SUP_X = np.vstack((SUP_X, SUP_0))
    
        # catch the case when estimated covariate > highest training covariate
        if covariate_estimates[i] > hi:
            print "Covariate {} > HI".format(covariate_estimates[i])
            for t, target in enumerate(info['PHI_targets'][-1]):
                model_PhiGamma = os.path.join(
                    config['PATH_TO_MODELS'], 
                    info['PHI_models'][-1][t])
                
                tmp_X, tmp_Y = implement_object_agnostic_synthesize(
                        config, 
                        object_features[i,:], 
                        model_PhiGamma)

                SUP_0[0,0]  = object_labels[i]
                SUP_0[0,1]  = tmp_Y
                SUP_0[0,2]  = target
                SUP_0[0,3]  = i
                    
                if PHI_X is None:
                    PHI_X = tmp_X
                    SUP_X = SUP_0
                else:
                    PHI_X = np.vstack((PHI_X, tmp_X))
                    SUP_X = np.vstack((SUP_X, SUP_0))  


    if verbose and not PHI_X is None:
        print 'Synthesized {} x {} RCNN features'.format(
            PHI_X.shape[0], PHI_X.shape[1])

    # return AGA synthesis result + metadata
    return PHI_X, SUP_X


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # configure cmdline parsing
    options = setup_parser().parse_args()

    # read system + AGA configurations
    sys_config = read_config(options.sys_config, verbose=options.verbose)
    aga_config = read_config(options.aga_config, verbose=options.verbose)

    # read image feature file (produced by Fast-RCNN)
    object_features, object_scores = read_RCNN_mat_file(options.input_file, options.verbose)
 
    if object_features is None:
        print "No features extracted!"
        sys.exit(-1)

    # allow no label file to be given (simply set label to 1 in that case)
    object_labels = None
    if options.label_file is None:
        object_labels = [1 for x in object_features.shape[0]]
    else:
        object_labels = sio.loadmat(options.label_file)['labels'].reshape(-1)
        object_labels = [label for label in object_labels]

    assert len(object_labels) == object_features.shape[0], 'Dimensionality mismatch!'

    # estimate attribute (strength) value
    covariate_estimates = object_agnostic_covariate_estimate(
        sys_config,
        aga_config,
        object_features,
        object_labels,
        options.verbose) 
    
    # run AGA-synthesis, given the estimate attribute value
    synthetic_features, metadata = object_agnostic_synthesize(
        sys_config,
        aga_config,
        object_features,
        object_labels,
        covariate_estimates,
        options.verbose)
 
    # unless AGA-synthesis does not produce anything, save results
    if not synthetic_features is None:
        sio.savemat(options.output_file, 
            {
                'CNN_feature': synthetic_features, 
                'CNN_metadata' : metadata
            })

if __name__ == '__main__':
    sys.exit( main() )
