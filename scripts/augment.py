"""
GOA - Guided Object Augmentation

rkwitt, mdixit, 2016
"""


import os
import sys
import yaml
import h5py
import scipy.io as sio
import subprocess
from optparse import OptionParser
import numpy as np


def setup_parser():
    """
    Setup the CLI parsing.
    """
    parser = OptionParser()
    parser.add_option("-c", "--config_file",    help="YAML config file.")
    parser.add_option("-i", "--input_file",     help="MATLAB input file with features + scores")
    parser.add_option("-y", "--information",     help="YAML information file.")
    parser.add_option("-o", "--output_file",    help="MATLAB output file with synthetic features + meta data")   
    parser.add_option("-s", "--source_object_file",    help="ASCII file with source object classes.")
    parser.add_option("-t", "--target_object_file",    help="ASCII file with target object classes.")
    parser.add_option("-g", "--gamma", default=0.7, type=float, help="Object detection threshold.")    
    parser.add_option("-a", action="store_true", default=False, dest="agnostic", help="Switch to agnostic")
    parser.add_option("-v", action="store_true", default=False, dest="verbose", help="Verbose output.")
    return parser


def read_file(file, verbose=False):
    """
    Read file line by line into list.
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


def collect_features(features, scores, object_names, skip_list, gamma=0.5, verbose=False):
    """
    tbd.
    """
    X = None # Returned features
    Y = []   # Returned object IDs for features

    O = len(object_names)
    assert O == scores.shape[1]

    for i in np.arange(O):        
        if object_names[i] in skip_list:
            continue
        pos = np.where(scores[:,i]>=gamma)[0]
        if pos.size>0:
            if X is None:
                X = features[pos,:]
            else:
                X = np.vstack((X, features[pos,:]))
            [Y.append(i) for x in pos]

    if verbose:
        print "Found {} object activations for {} object(s)".format(
            X.shape[0], len(np.unique(Y)))
        print [object_names[x] for x in Y]

    return X, Y


def implement_object_covariate_estimate(config, X, model):
    """
    tbd.
    """
    tmp_data_file = os.path.join( config['TEMP_DIR'], 'X.hdf5')
    tmp_pred_file = os.path.join( config['TEMP_DIR'], 'prediction.hdf5')

    with h5py.File( tmp_data_file, 'w') as hf:
        if len(X.shape) == 1: # only one entry 
            X = X.reshape((X.shape[0],1))
        hf.create_dataset('/X', data=X) 
    
    cmd = [
        config['TORCH'],
        config['TEST_COR'],
        '-modelCOR', model,
        '-dataFile', tmp_data_file,
        '-outputFile', tmp_pred_file]
    print cmd
    subprocess.call( cmd )
    
    vals = None
    with h5py.File( tmp_pred_file ,'r') as hf:
        vals = np.asarray( hf.get('Y_hat')).reshape(-1)
    return vals


def object_covariate_estimate(config, info, object_features, object_labels, object_names, verbose=False):
    """
    tbd.
    """
    vals = []
    N = object_features.shape[0]
    assert N == len(object_labels)

    for i in np.arange(N):
        object_name = object_names[object_labels[i]]
        model_COR = os.path.join(
            config['PATH_TO_MODELS'], 
            object_name, 
            info[object_name]['COR_object_model'])

        tmp_val = implement_object_covariate_estimate(
            config, 
            object_features[i,:], 
            model_COR)
        [vals.append(x) for x in tmp_val]
        
    return vals


def implement_object_agnostic_covariate_estimate(config, X, model):
    """
    tbd.
    """
    tmp_data_file = os.path.join( config['TEMP_DIR'], 'X.hdf5')
    tmp_pred_file = os.path.join( config['TEMP_DIR'], 'prediction.hdf5')

    with h5py.File( tmp_data_file, 'w') as hf:
        if len(X.shape) == 1: # only one entry 
            X = X.reshape((X.shape[0],1))
        hf.create_dataset('/X', data=X) 
    
    cmd = [
        config['TORCH'],
        config['TEST_COR'],
        '-modelCOR', model,
        '-dataFile', tmp_data_file,
        '-outputFile', tmp_pred_file]
    print cmd
    subprocess.call( cmd )

    vals = None
    with h5py.File( tmp_pred_file ,'r') as hf:
        vals = np.asarray( hf.get('Y_hat')).reshape(-1)
    return vals


def object_agnostic_covariate_estimate(config, object_features):
    vals = None
    N = object_features.shape[0]
    
    for i in np.arange(N):
        model_COR = os.path.join(config['PATH_TO_MODELS'], 'agnosticCOR.t7')
        vals = implement_object_agnostic_covariate_estimate(
            config, 
            object_features[i,:], 
            model_COR)
    return vals


def implement_object_synthesize(config, X, model):
    """
    tbd.
    """
    tmp_data_file = os.path.join( config['TEMP_DIR'], 'X.hdf5')
    tmp_pred_file = os.path.join( config['TEMP_DIR'], 'X_hat.hdf5')
    
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
    print cmd
    subprocess.call( cmd )
   
    X_hat = None # Synthesized RCNN feature
    Y_hat = None # Covariate as predicted for the synthesized feature
    with h5py.File( tmp_pred_file ,'r') as hf:
        X_hat = np.asarray( hf.get('X_hat'))
        Y_hat = np.asarray( hf.get('Y_hat_EDNCOR'))
    return X_hat, Y_hat


def object_synthesize(config, info, object_features, object_labels, object_names, estimated_covariates, verbose=False):
    EDN_X = None
    EDN_0 = np.zeros((1,4096))
    SUP_X = None
    SUP_0 = np.zeros((1,4))

    N = object_features.shape[0]
    assert N == len(object_labels)

    # Iterate over all RCNN activations
    for i in np.arange(N):
        label = object_labels[i]
        object_name = object_names[label]
        covariate = estimated_covariates[i]

        # Iterate over all covariate intervals for which we have models 
        for j, interval in enumerate(info[object_name]['intervals']):

            # Interval = [lo,hi]
            lo = interval[0]
            hi = interval[1]

            # If the estimated covariate is in an interval for which we
            # have a model, call the model to synthesize actiavations for
            # a collection of covariate targets.
            if (covariate >= lo and covariate <= hi):

                covariate_targets = info[object_name]['EDN_targets'][j]    

                for m, EDN_model in enumerate(info[object_name]['EDN_object_models'][j]):

                    # The object-specific EDN model is in the object folder
                    model_EDNCOR = os.path.join(
                        config['PATH_TO_MODELS'], 
                        object_name, 
                        EDN_model)

                    tmp_X, tmp_Y = implement_object_synthesize(
                        config, 
                        object_features[i,:], 
                        model_EDNCOR)

                    SUP_0[0,0]  = label                 # Store object label ID
                    SUP_0[0,1]  = tmp_Y                 # Store output of EDN+COR on synthesized activation                
                    SUP_0[0,2]  = covariate_targets[m]  # Store the covariate target that was used            
                    SUP_0[0,3]  = i                     # Store the object ID within that image          
                    if EDN_X is None:
                        EDN_X = tmp_X
                        SUP_X = SUP_0
                    else:
                        EDN_X = np.vstack((EDN_X, tmp_X))
                        SUP_X = np.vstack((SUP_X, SUP_0))
                       

    if verbose and not EDN_X is None:
        print 'Synthesized {}x{} RCNN features'.format(
            EDN_X.shape[0], EDN_X.shape[1])

    return EDN_X, SUP_X


def main(argv=None):
    if argv is None:
 	    argv = sys.argv

    parser = setup_parser()
    (options, args) = parser.parse_args()

    # 1. Read system config information
    config = read_config(
	   options.config_file, 
       verbose=options.verbose)

    fid = open(options.information)
    info = yaml.load(fid)
    fid.close()

    # 2. Read source/target object_names (could be the same)
    source_object_names = read_file(
        options.source_object_file,
        verbose=options.verbose)
    target_object_names = read_file(
        options.target_object_file,
        verbose=options.verbose)

    # 3. Load CNN features + scores from MATLAB .mat file
    all_features, all_scores = read_RCNN_mat_file(
        options.input_file,
        options.verbose)

    # 4. Collect relevant object features from image
    skip_list = ['__background__', 'others']
    object_features, object_labels = collect_features(
        all_features,           # All RCNN activations
        all_scores,             # All RCNN scores
        target_object_names,    # Names of objects corresponding to score columns
        skip_list,              # Skip objects
        options.gamma,          # Detection threshold
        options.verbose)        # Verbose output

    if options.agnostic:
        pass        

    else:
        covariate_estimates = object_covariate_estimate(
            config,                 # System config
            info,                   # Model information
            object_features,        # RCNN activations for scores>gamma
            object_labels,          # RCNN object labels for activations
            target_object_names,    # Names of objects corresponding to score columns
            options.verbose)        # Verbose output

        synthetic_features, metadata = object_synthesize(
            config,
            info,
            object_features,
            object_labels,
            target_object_names,
            covariate_estimates,    # Estimates for the covariate (e.g., depth)
            options.verbose)

        if not synthetic_features is None:
            sio.savemat(options.output_file, 
                {
                    'CNN_feature': synthetic_features, 
                    'CNN_metadata' : metadata
                })


if __name__ == '__main__':
    sys.exit( main() )