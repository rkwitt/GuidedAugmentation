import subprocess
import numpy as np
import sys
import os


SUNRGBD_common 	= '/scratch2/rkwitt/Mount/images/objects'

TORCH 			= '/home/pma/rkwitt/torch/install/bin/th'
TRAIN_PRE 		= '/home/pma/rkwitt/GuidedAugmentation/pretrain.lua'
TRAIN_BIN 		= '/home/pma/rkwitt/GuidedAugmentation/adjuster.lua'
MODEL 			= '/home/pma/rkwitt/GuidedAugmentation/torch/autoencoder.lua'


class meta:
		def __init__( self ):
			self.name = None		# Object name
			self.lo = None			# Object covariate lo-val
			self.hi = None			# Object covariate hi-val
			self.targets = []		# Covariate values for regression
			self.train_file = None	# Training file
			self.cv_files = []		# Crossvalidation files


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

	meta_obj.name = parts[0]
	meta_obj.lo, meta_obj.hi = float( parts[1] ), float( parts[2] )
	[meta_obj.cv_files.append( x ) for x in parts[3:-2]]
	meta_obj.train_file = parts[-1]

	return meta_obj


def set_covariate_targets( meta_objs, range=np.arange(1, 5, 0.5)):
	"""Set covariate targets based on covariate data"""

	for meta_obj in meta_objs:
		for r in range:
			if r >= meta_obj.lo and r <= meta_obj.hi:
				continue
			meta_obj.targets.append( r )


def trainer( meta_objs, pretrain=False ):

	for meta_obj in meta_objs:

		obj_path = os.path.join( SUNRGBD_common, meta_obj.name )

		if pretrain:

			cmd = [TORCH,
				TRAIN_PRE,
				'-dataFile',  os.path.join(obj_path, 'train.hdf5'),
				'-saveModel', os.path.join(obj_path, 'model_pretrain.t7'),
				'-modelFile', MODEL,
				'-cuda']
			print cmd
			subprocess.call( cmd )


		for target in meta_obj.targets:

			print "%20s | [%.4f, %.4f] -> %.4f" % (
				meta_obj.name,
				meta_obj.lo,
				meta_obj.hi,
				target)


def main():

	data = {}

	# read file and process line-by-line
	for line in read_file( sys.argv[1] ):

		# parse entry into meta object
		meta_obj = parse_entry( line )

		obj_name = meta_obj.name

		if not obj_name in data:
			data[obj_name] = []

		data[obj_name].append( meta_obj )

	for obj_name in data:
		set_covariate_targets( data[obj_name] )

	for obj_name in data:
		trainer( data[obj_name], pretrain=True )


if __name__ == "__main__":
    main()
