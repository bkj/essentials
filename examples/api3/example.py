import numpy as np
import pyCUREVERSE as rv

for _ in range(10):
  for reverse, dtype in [(rv.reverseI, np.int32), 
                        (rv.reverseL, np.int64),
                        (rv.reverseF, np.float32),
                        (rv.reverseD, np.float64)]:

      X = np.arange(100, dtype=dtype)
      reverse(X)
      
      print(X)


