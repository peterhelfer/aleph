#!/usr/bin/env python
#
"""
entropy.py
~~~~~~

Fooling around with entropy

By Peter Helfer, 2023
"""

import math
import numpy as np

fair_die = (1/6, 1/6, 1/6, 1/6, 1/6, 1/6)
loaded_die = (1/50, 1/50, 1/50, 1/50, 1/50, 9/10)

def entropy(outcome_probs):
    return sum(map(lambda x : x * math.log2(1/x),outcome_probs))

print('Fair die   entropy: {:.2f}'.format(entropy(fair_die)))
print('Loaded die entropy: {:.2f}'.format(entropy(loaded_die)))
      
