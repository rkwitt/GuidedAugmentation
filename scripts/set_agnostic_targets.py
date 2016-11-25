import subprocess
import tempfile
import numpy as np
import yaml
import h5py
import sys
import os

data = None
info = None

def set_EDN_targets(info, range=np.arange(1, 5, 0.5)):
    
    for interval in info['intervals']:

        lo, hi = interval
        targets = []
        
        for r in range:
            if r >= lo and r <= hi:
                continue
            targets.append(r)

        info['EDN_targets'].append(tuple(targets))


with open( sys.argv[1] ) as fid:
    data = fid.readlines()
    data = [l.rstrip() for l in data]

for line in data:
    
    parts = line.split(',')

    lo = float(parts[0])
    hi = float(parts[1])

    EDN_train_file = parts[-1]

    if info is None:
        info = {
            'intervals':            [],                 # [l_i, h_i]
            'EDN_train_files' :     [],                 # Training data file for EDN
            'EDN_pre' :             'EDN_pre.t7',       # Name of pretrained EDN model
            'AR_agnostic_model':    'agnosticAR.t7',    # Name of pretrained AR model
            'EDN_models':           [],                 # Will hold the final trained EDN models
            'EDN_targets' :         []                  # Will hold the attribute target values for the EDN models
            }

    info['intervals'].append((lo,hi))
    info['EDN_train_files'].append(EDN_train_file)

# DEPTH
#set_EDN_targets(info)
# POSE
set_EDN_targets(info, np.arange(np.deg2rad(45), np.pi, np.deg2rad(25)))

fid = open(sys.argv[2], 'w')
yaml.dump(info, fid)
fid.close()
