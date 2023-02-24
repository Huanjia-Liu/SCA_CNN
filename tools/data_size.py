import h5py
import numpy as np
import sys

sys.path.append("../")

import hyperparam

path = hyperparam.hyperparam.path

with h5py.File(path,'r') as f:
    print(f['traces'].shape)
