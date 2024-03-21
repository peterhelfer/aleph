#!/usr/bin/env python
#

import numpy as np

a = np.array([0,0,1,0,0])
print(a)
print(np.equal.outer(a,a).astype(int))



