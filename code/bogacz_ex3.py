#!/usr/bin/env python
#
"""
bogacz_ex3.py
~~~~~~

Exercise 3 from Bogacz 2017
"""

import numpy as np
import matplotlib.pyplot as plt

# Weights
#
v_p = 3     # Prior mean object size
var_p = 1   # (prior) variance in v_p
var_u = 1   # variance in dependency of u on v 

# Nodes:
#
u = 2          # observed object brightness
v = v_p        # inferred (posterior) object size

e_p = 0        # prediction error in v
e_u = 0        # prediction error in g(v) = v^2


start_time = 0
end_time = 5
dt = 0.01       # time step
n_steps = int((end_time - start_time) / dt + 1)   # number of steps

# Arrays for accumulating variable values

t_acc   = np.zeros(n_steps)
e_u_acc = np.zeros(n_steps)
e_p_acc = np.zeros(n_steps)
v_acc = np.zeros(n_steps)

# Let's boogie!
#
t = 0
for i in range(n_steps):
    e_u = e_u + dt * (u - v * v - var_u * e_u)
    v =   v   + dt * (e_u * 2 * v - e_p)
    e_p = e_p + dt * (v - v_p - e_p * var_p)
    #print(f'{i:02.2f}  {t:.2f} {e_u:.2f} {e_p:.2f} {u:.2f} {v:.2f}')

    t_acc[i]   = t
    e_u_acc[i] = e_u
    e_p_acc[i] = e_p
    v_acc[i]   = v
    t = t + dt

# Plot the results
#
fig, ax = plt.subplots()
ax.plot(range(n_steps), e_u_acc)
ax.plot(range(n_steps), e_p_acc)
ax.plot(range(n_steps), v_acc)

ax.set(xlabel='Time', 
       title='Bogacz 2017, Exercise 3')
ax.grid()
ax.legend(['e_u, p_u, v'])
plt.show()
