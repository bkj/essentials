"""
  sssp_test.py
"""

import gunrock_sssp
import numpy as np
from scipy import sparse

# --
# Random graph

csr = sparse.random(100, 100, density=0.1)
csr = (csr + csr.T) > 0
csr.data = csr.data.astype(np.float32)

# --
# Run

single_source = 0
distances     = gunrock_sssp.run_sssp_IIF(single_source, csr.indptr, csr.indices, csr.data)
print(distances)