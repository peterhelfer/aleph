#!/usr/bin/env python
#
# First predictive coding attempt:
# Classify static 105x105 images (rao_1999 Section 7.1)

import numpy as np
import util as ut

R_MAX = 5 # Size of state vector r
I_MAX_X = 105 # X dimension of input images
I_MAX_Y = 105 # Y dimension of input images
I_MAX = I_MAX_X * I_MAX_Y # Size of flattened input image

rng = np.random.default_rng()

# We implement the following equations from Rao 1999:
#   Eqs. 18-19 to learn U and V (except that V is a fixed identity matrix when
#              learning static images, as explained in section 7.1)
#   Eqs. 20-21 to compute r_hat and r_prime, with the simplified formulas for
#              r_bar and r, as described in section 5.4.

### Initialize ###

r = np.zeros([R_MAX]) # The state vector
U = rng.random([I_MAX, R_MAX]) # Generative matrix (model of the measurement matrix)
V = rng.identity(R_MAX) # Prediction matrix (model of the state transition matrix)
                        # For static images, it's an unchanging identity matrix

### Train ###

# Load the training images

### Test##



"""
dt = 1 # simulation timestep

# The state vector x and its covariance matrix P
#
x = np.array([0, 1])           # Initial estimate: position and velocity
P = np.array([[0.1, 0], [0, 0.1]]) # Initial estimate: zero

# The prediction matrix F (model kinematics: x(t) = Fx(t-1) + Bu(t))
#
F = np.array([[1, dt], [0, 1]])    # x <-- x + v*dt;  v <-- v

# The Control vector u and Control matrix B
#
u = np.array([0, 0])    # control vector: u[1] is acceleration
B = np.array([0, dt])   # control matrix: v = v + a*dt

# The measurement z and its covariance matrix R
#
z = np.zeros_like(x)               # position and velocity based on the sensors
R = np.array([[0.1, 0], [0, 0.1]]) # based on our understanding of the sensors  


# The sensor matrix H - predicts a measurement from a model state
#                       (In other word, the model's idea of how the sensors
#                       produce a measurement from the ground truth!)
# 
H = np.array([[1, 0], [0, 1]])  # Keeping it simple here


# Run the model for t_max timesteps

t_max = 10

for t in range(t_max):

    # Prediction: Run the model kinematics
    #
    u = np.array([0,0])  # set u[1] to desired acceleration
    x = np.matmul(F, x) + np.matmul(B, u) # prediction
    P = np.matmul(np.matmul(F, P), F.T) # Update state uncertainty

    # Update: Weighted average of prediction and measurement
    #
    z = x  # Get the new measurement

    # Calculate the Kalman gain K. It represents the weight that will be given to
    # the new measurement when calculating a weighted average of the model prediction
    # and the new measurement, based on their respective uncertainties, i.e.
    # covariance matrices.
    #
    K = np.matmul(P,np.matmul(H.T,np.linalg.inv(np.matmul(H,np.matmul(P,H.T))+R)))

    # Calculate the new model state and update the covariance matrix
    #
    x = x + np.matmul(K, (z - np.matmul(H, x)))
    P = P - np.matmul(K, np.matmul(H, P))


    #print(x, ut.fmt_seq('{:5.1f}', P))
    print(x)
    print(P)
    print()
"""
