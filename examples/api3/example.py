import test
import ctypes

import numpy as np

x = np.arange(1, 5).astype(np.int32)
y = np.arange(1, 10).astype(np.int32)
res = test.do_test_ssspI(x, y)
print(res)

x = np.arange(1, 5).astype(np.float32)
y = np.arange(1, 10).astype(np.float32)
res = test.do_test_ssspF(x, y)
print(res)