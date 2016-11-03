import os
import sys
import pickle
import h5py
import numpy as np

from meta import meta

SUNRGBD_common  = '/scratch2/rkwitt/Mount/images/objects'

info = pickle.load( open( sys.argv[1], 'r' ) )

for object_name in info:

	obj_path = os.path.join( SUNRGBD_common, object_name )
	
	for sub_obj in info[object_name]:

		for t, target in enumerate( sub_obj.covariate_targets ):

			target_mean = []

			for cv_feature_file in sub_obj.feature_regression_cv_files:

				prediction_file = os.path.splitext( cv_feature_file )[0] + '_' + str( target ) + '_prediction.hdf5'
	
				with h5py.File( os.path.join( obj_path, prediction_file ), 'r') as hf:
					pred_ae_Y = np.asarray( hf.get('pred_ae_Y') )
					target_mean.append( pred_ae_Y.mean() )

			# print per target statistics for that object
			print '{:20s} | {}-{} | Target: {:4f} | ED-pred.: {:4f} +/- {:4f} [Err: {:4f}]'.format(
				object_name, 
				sub_obj.covariate_lo, 
				sub_obj.covariate_hi, 
				target, 
				np.asarray(target_mean).mean(), 
				np.asarray(target_mean).std(),
				np.abs(target-np.asarray(target_mean).mean())
				)

    			
    			