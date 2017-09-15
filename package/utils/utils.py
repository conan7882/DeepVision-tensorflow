import os
from datetime import datetime

import numpy as np

__all__ = ['get_rng']

_RNG_SEED = None

def get_rng(obj = None):
	seed = (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
	if _RNG_SEED  is not None:
		seed = _RNG_SEED 
	return np.random.RandomState(seed)