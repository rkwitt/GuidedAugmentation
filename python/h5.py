import sys
import h5py

f = h5py.File(sys.argv[3], "w")

import sys
import numpy as np

X = np.load(sys.argv[1])
S = np.loadtxt(sys.argv[2])

data = f.create_dataset("data", data=X)
meta = f.create_dataset("meta", data=S)

f.close()

with h5py.File(sys.argv[3],'r') as f:
    data = f.get('data')
    print data.shape
