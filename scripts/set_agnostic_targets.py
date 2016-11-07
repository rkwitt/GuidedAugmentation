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
            'intervals':            [], 
            'EDN_train_files' :     [],
            'EDN_pre' :             'EDN_pre.t7',
            'COR_agnostic_model':   'agnosticCOR.t7',
            'EDN_models':           [],
            'EDN_targets' :         []}

    info['intervals'].append((lo,hi))
    info['EDN_train_files'].append(EDN_train_file)

set_EDN_targets(info)

fid = open(sys.argv[2], 'w')
yaml.dump(info, fid)
fid.close()