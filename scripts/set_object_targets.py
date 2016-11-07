import subprocess
import tempfile
import numpy as np
import yaml
import h5py
import sys
import os

data = None
info = dict()

def set_EDN_targets(info, range=np.arange(1, 5, 0.5)):
    
    for obj in info:
        for interval in info[obj]['intervals']:

            targets = []
            lo, hi = interval

            for r in range:
                if r >= lo and r <= hi:
                    continue
                targets.append(r)

            info[obj]['EDN_targets'].append(tuple(targets))


with open( sys.argv[1] ) as fid:
    data = fid.readlines()
    data = [l.rstrip() for l in data]

for line in data:
    
    parts = line.split(',')

    obj = parts[0]

    if not obj in info:
        info[obj] = { 
            'intervals': [], 
            'EDN_cv_files': [], 
            'EDN_train_files': [],
            'COR_object_model': 'objectCOR.t7',
            'EDN_targets': [],
            'EDN_pre': 'EDN_pre.t7', 
            'EDN_object_models': []}

    lo = float(parts[1])
    hi = float(parts[2])
    EDN_train_file = parts[-1]
    EDN_cv_files = parts[3:-1]

    info[obj]['intervals'].append((lo,hi))
    info[obj]['EDN_cv_files'].append(tuple(EDN_cv_files))
    info[obj]['EDN_train_files'].append(EDN_train_file)

set_EDN_targets(info)

fid = open(sys.argv[2], 'w')
yaml.dump(info, fid)
fid.close()

# DEBUG
# fid = open(sys.argv[2], 'r')
# test = yaml.load(fid)
# fid.close()
# print(test['chair']['intervals'][4])
# print(test['chair']['EDN_cv_files'][4])
# print(test['chair']['EDN_targets'][4])





