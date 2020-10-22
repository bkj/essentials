"""
  sssp_test.py
"""

import gunrock_sssp
import numpy as np
from scipy import sparse

# --
# Random graph

x = sparse.random(100, 100, density=0.1)
x = (x + x.T) > 0
x.data = x.data.astype(np.float32)

# --
# Run

for single_source in range(10):
  res = gunrock_sssp.do_sssp_IIF(single_source, x.indptr, x.indices, x.data)
  print(res)