#!/usr/bin/env python
#
"""
aleph_0.py
~~~~~~

The aleph_0 network
"""

#-------------------------------------------------------------------------------
# Copyright (C) 2023  Peter Helfer
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <https://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import sys
import scipy.stats
from scipy.special import softmax
import numpy as np

import util as ut

# Command-line argument parser
#
def get_arg_parser():
    parser = ut.TerseArgumentParser()
    # General utility args
    #
    parser.add_str_arg2('t', 'trace_level',     'INFO2', 'Trace level, one of [FLOW,DEBUG,INFO3,INFO2,INFO,WARN,ERROR,FATAL]')


    # Sweep these hyperparameters to optimize the separation property
    #
    parser.add_float_arg('p_r_input',           0.20,   'Fraction of recurrents that receive input')
    parser.add_float_arg('p_ir_conn',           0.05,   'Fraction of input-recurrent connections to activate')
    parser.add_float_arg('ir_min_w',            0.0,    'Minimum initial weight for ir connections')
    parser.add_float_arg('ir_max_w',            1.0,    'Maximum initial weight for ir connections')
    parser.add_float_arg('rr_min_w',            0.0,    'Minimum initial weight for rr connections')
    parser.add_float_arg('rr_max_w',            1.0,    'Maximum initial weight for rr connections')
    parser.add_float_arg('tau_mem',             40,     'Time constant of membrane leak current in recurrent')
    parser.add_float_arg('c_mem',               1.44,   'Membrane capacitance of recurrent')
    parser.add_float_arg('r_base_thresh',       4.0,    "Base (unbiased) spiking threshold of recurrents' membrane potential")
    parser.add_float_arg('refr_period_msec',    5.825,  'Refractory period, msec')
    
    # Using "standard" values for these (although we could sweep them)
    #
    parser.add_float_arg('p_rr_conn_ee',    0.3,    'Prob. of h-h conn. btwn EE neurons')
    parser.add_float_arg('p_rr_conn_ei',    0.2,    'Prob. of h-h conn. btwn EI neurons')
    parser.add_float_arg('p_rr_conn_ie',    0.4,    'Prob. of h-h conn. btwn IE neurons')
    parser.add_float_arg('p_rr_conn_ii',    0.1,    'Prob. of h-h conn. btwn II neurons')

    parser.add_float_arg('lambda_p_rr_conn',    1.7,    'Spatial decay rate of h-h connection prob.')

    parser.add_float_arg('p_r_inhib',           0.20,   'Prob that a recurrent is inhibitory')

    parser.add_float_arg('r_ee_w',              9.791,     'weight of an EE connection in the recurrent layer')
    parser.add_float_arg('r_ei_w',              3.338,     'weight of an EI connection in the recurrent layer')
    parser.add_float_arg('r_ie_w',              5.062,     'weight of an IE connection in the recurrent layer')
    parser.add_float_arg('r_ii_w',              3.624,     'weight of an II connection in the recurrent layer')

    # Sweep these hyperparams to optimize classification
    #
    parser.add_float_arg('max_init_ho_w',       0.05,    'Max initial weight of recurrent-output connection')
    parser.add_float_arg('learn_rate',          1.0,     'Learning rate for the output weights')
    parser.add_int_arg('num_epochs',            1,       'Number of epochs')

    # Clock intervals
    #
    parser.add_int_arg('input_intrvl',          1,       'Length of frames in the input, msec')
    parser.add_int_arg('clock_intrvl',          1,       'Simulation clock interval, msec')

    return parser

args = {}

# Utility functions
#
def sigmoid(x):
    """The sigmoid function."""
    if np.min(x) < -500 or np.max(x) > 500:
        return np.zeros_like(x) # Avoid under/overflow
    return 1.0 / (1.0 + np.exp(-x))

# The Network
#
class Network:
    def __init__(self, name, num_inputs, num_subnets, subnet_size, rng, args):
        """Create network with the specified dimensions.
        :param name: Network name, used in trace messages
        :param num_inputs: Number of input neurons (channels)
        :param num_subnets: Number of subnetworks
        :param subnet_size: Number of units in each subnet
        :param rng: Random number generator to use for weights, connectivity, etc.
        :param args: dictionary of cmdline args
        """
        self.name               = name
        self.num_inputs         = num_inputs
        self.num_subnets        = num_subnets
        self.subnet_size        = subnet_size
        self.rng                = rng

        self.clock_intrvl       = args.clock_intrvl

        self.ir_min_w           = args.ir_min_w
        self.ir_max_w           = args.ir_max_w
        self.rr_min_w           = args.rr_min_w
        self.rr_max_w           = args.rr_max_w
        self.tau_mem            = args.tau_mem
        self.c_mem              = args.c_mem
        self.r_base_thresh      = args.r_base_thresh

        self.input_intrvl       = args.input_intrvl
        self.learn_rate         = args.learn_rate
        self.max_init_ho_w      = args.max_init_ho_w
        self.p_r_inhib          = args.p_r_inhib
        self.p_ir_conn          = args.p_ir_conn
        self.refr_period_msec   = args.refr_period_msec
     

        ut.info2(f'Creating network ni={self.num_inputs}, ' +
                f'nsn={self.num_subnets}, snz={self.subnet_size}')

        # Attributes of input (i_), recurrent (r_), output (o_) and
        # neuromodulator (m_) neurons:
        #   ic: input current
        #   rc: recurrent current
        #   oc: current from output neuron
        #   v:  membrane potential
        #   v_thresh: spiking threshold
        #   s:  spike == 1 when spiking, 0 otherwise
        #   refr: refractory period countdown timer
        #   sign: +1 for excitatory, -1 for inhibitory neurons

        # Input neurons
        #
        self.i_s  = np.zeros(self.num_inputs)

        # Recurrent neurons - Each subnet will have subnet_size neurons, one "output" neuron that
        # reflects subnet activity level, and a neuromodulation neuron that other subnets can
        # excite or inhibit.
        # 
        #
        self.num_recurrents = self.num_subnets * self.subnet_size
        self.r_ic       = np.zeros(self.num_recurrents)
        self.r_rc       = np.zeros(self.num_recurrents)
        self.r_v        = np.zeros(self.num_recurrents)
        self.r_v_thresh = np.full(self.num_recurrents, self.r_base_thresh)
        self.r_s        = np.zeros(self.num_recurrents)
        self.r_refr     = np.zeros(self.num_recurrents)
        self.r_sign     = np.ones(self.num_recurrents)

        # output neurons
        #
        self.o_a = np.zeros(int(self.num_subnets))

        # neuromodulator neurons - TODO: figure out how these will work. How does a subnet know
        # which other subnets to modulate, and how does it modulate selectively?
        #
        self.m_rc       = np.zeros(self.num_subnets)
        self.m_v        = np.zeros(self.num_recurrents)
        self.m_s        = np.zeros(self.num_recurrents)

        # weight matrices
        #
        self.ir_w = np.zeros((self.num_recurrents, self.num_inputs))
        self.rr_w = np.zeros((self.num_recurrents, self.num_recurrents))
        self.ro_w = np.zeros((self.num_subnets, self.num_recurrents + 1)) #incl. bias unit

        # Randomly make some recurrent neurons inhibitory
        #
        self.r_sign = rng.choice((-1, 1), size=self.num_recurrents, p=(self.p_r_inhib, 1-self.p_r_inhib))

        # Randomize the weights from input to recurrent neurons
        #
        self.ir_w = (rng.random((self.num_recurrents, self.num_inputs)) *
                     (self.ir_max_w - self.ir_min_w) + self.ir_min_w)

        # Initialize the weights for each subnet
        #
        for sn in range(self.num_subnets):
            n = self.subnet_size
            # identify the recurrent neurons
            lo = sn * n
            hi = lo + n

            # connect them with random weights
            self.rr_w[lo:hi, lo:hi] = (rng.random((n, n)) *
                                       (self.rr_max_w - self.rr_min_w) + self.rr_min_w)

        """
        # Create the recurrent-recurrent weight matrix
        #
        ut.info2('connecting recurrents')

        # Create the recurrent-recurrent connections
        #
        exc = (self.r_sign >= 0).astype(int) # Indices of excitatory neurons
        inh = (self.r_sign <  0).astype(int) # Indices of inhibitory neurons

        ee_conns = np.outer(exc, exc) # excitatory-to-excitatory connections
        ei_conns = np.outer(exc, inh) # etc.
        ie_conns = np.outer(inh, exc)
        ii_conns = np.outer(inh, inh)

        np.fill_diagonal(ee_conns, 0) # no self-connects
        np.fill_diagonal(ei_conns, 0)
        np.fill_diagonal(ie_conns, 0)
        np.fill_diagonal(ii_conns, 0)

        # Set the probability of connecting each pair of neurons,
        # using excitatory-to-excitatory connection probability, etc.

        p = (ee_conns * self.p_hh_conn_ee +
             ei_conns * self.p_hh_conn_ei +
             ie_conns * self.p_hh_conn_ie +
             ii_conns * self.p_hh_conn_ii)


        # Decide which recurrents to connect, based on the probabilities
        #
        connect = rng.random((len(locs), len(locs))) < p # Random p fraction of connections
        
        # Connect them, using weights based on exc/inh
        #
        self.rr_w = (ee_conns * self.r_ee_w +
                     ei_conns * self.r_ei_w +
                     ie_conns * self.r_ie_w +
                     ii_conns * self.r_ii_w) * connect
        """

        ut.info2('subnets initialized')

        # Objects for recording neuron states
        #
        self.r_s_acc = None
        self.r_v_acc = None

    def init_recording(self):
        """Start accumulating neuron spike states"""
        self.r_s_acc = np.zeros(shape=(0, self.r_s.size))        
        self.r_v_acc = np.zeros(shape=(0, self.r_v.size))        

    def get_recording(self):
        """Get recorded neuron states"""
        return Recording(self.r_s_acc, self.r_v_acc)

    def stop_recording(self):
        """Stop accumulating neuron spike states"""
        self.r_s_acc = None
        self.r_v_acc = None
        
    def clear(self):
        """Restore the network to pristine state (but don't reinitialize the output
        weights)"""
        # Clear the input neuron spikes
        self.i_s.fill(0)
        # Clear the recurrent neuron currents, voltages,
        # spikes, refr, and xtrace values
        self.r_ic.fill(0)
        self.r_rc.fill(0)
        self.r_v.fill(0)
        self.r_s.fill(0)
        self.r_refr.fill(0)
        
        # Clear the output neuron activity levels
        self.o_a.fill(0)

    def load_sample(self, sample):
        """Prepare a sample for processing and reset the clock to zero"""
        # Transpose the sample so that we can step through all input
        # channels in parallel
        self.sample = sample
        self.clock = 0
        self.input_counter = 0
        self.next_input_time = 0

    #@profile
    def tick(self):
        """Do processing for one clock tick, then advance the clock"""
        #ut.info3(f'Tick: {self.clock}')
        # Read inputs from the sample
        if self.clock >= self.next_input_time:
            if self.input_counter >= len(self.sample):
                self.done = True
                return
            # Read next set of input values
            self.i_s = self.sample[self.input_counter]

            # Update the input currents to the recurrent layer
            self.r_ic = np.dot(self.ir_w, self.i_s)

            # Set up for next input
            self.input_counter += 1
            self.next_input_time += self.input_intrvl

        # Update the recurrent (spike) currents
        self.r_rc = np.dot(self.rr_w, (self.r_s * self.r_sign))

        # Terminate all spikes (TODO: configurable spike duration)
        self.r_s.fill(0)


        # Calculate total input currents to recurrents
        c = (-self.r_v / self.tau_mem +  # leak
             (self.r_ic + self.r_rc) / self.c_mem) # accumulation
        c = c.clip(min=0, max=None, out=self.r_v) # keep it non-negative

        # Update membrane potentials of recurrents
        self.r_v += self.clock_intrvl * c

        # Update spiking states of the recurrents
        self.r_s = np.logical_and((self.r_refr == 0),
                                  (self.r_v > self.r_v_thresh))
        #ut.info3(f'{self.clock} Spikes: {np.count_nonzero(self.r_s)}')

        # If recording, accumulate neuron spiking and membrane potential arrays
        #
        if self.r_v_acc is not None:
            self.r_v_acc = np.vstack((self.r_v_acc, self.r_v))
        if self.r_s_acc is not None:
            self.r_s_acc = np.vstack((self.r_s_acc, self.r_s))

        # Count down refractory periods
        #
        self.r_refr = np.maximum(self.r_refr - self.clock_intrvl, 0)
        
        # Clear membrane potential of spiking neurons and start refractory
        # period
        #
        self.r_v[self.r_s!=0] = 0
        self.r_refr[self.r_s!=0] = self.refr_period_msec


        # Advance the clock
        self.clock += self.clock_intrvl

    def process_sample(self):
        """Run the network until all the input has been processed.
        :return: The output activations"""

        self.done = False
        while not self.done:
            self.tick()

    def forward_pass(self, sample):
        """Load a sample and execute a forward pass
        and readout network.
        :param sample: A sample
        :return: The resulting output activations
        """
        self.clear()
        self.load_sample(sample)
        self.process_sample()
        return self.o_a
        
        
    def dump_topology(self):
        """Print network topology info (for debugging)"""
        nz = np.count_nonzero(self.ir_w)
        sz = np.size(self.ir_w)
        ut.info2(f'{self.name} nonzero(ir_w): {nz}/{sz} = {nz/sz}')
        nz = np.count_nonzero(self.rr_w)
        sz = np.size(self.rr_w)
        ut.info2(f'{self.name} nonzero(rr_w): {nz}/{sz} = {nz/sz}')
        nz = np.count_nonzero(self.ro_w)
        sz = np.size(self.ro_w)
        ut.info2(f'{self.name} nonzero(ro_w): {nz}/{sz} = {nz/sz}')
        nz = np.count_nonzero(self.r_sign == -1)
        sz = np.size(self.r_sign)
        ut.info2(f'{self.name} inhib(h): {nz}/{sz} = {nz/sz}')
        #ut.info2(f'ir_w:\n{self.ir_w}')
        #ut.info2(f'rr_w:\n{self.rr_w}')

    def dump_connectivity(self):
        """Print the number of non-zero conn and 
        sum of ir_w and rr_w for all recurrents"""
        for h in range(self.num_recurrents):
            ut.info2(f'{h:2d} ir nz:{np.count_nonzero(self.ir_w[h])} ' +
                    f'{np.sum(self.ir_w[h])} ' +
                    f'ir nz:{np.count_nonzero(self.rr_w[h])} ' 
                    f'{np.sum(self.rr_w[h])}')

    def dump_state(self):
        """Print network state info (for debugging)"""
        #ut.info2("i_s:', self.i_s)
        #ut.info2('r_ic:', self.r_ic)
        #ut.info2('r_rc:', self.r_rc)
        #ut.info2('r_v:', fmt_seq('{:5.3g}', self.r_v))
        #ut.info2('r_s:', self.r_s)
        #ut.info2('r_sign:', self.r_sign)
        ut.info2('r_xtrace:', self.r_xtraces)
        #ut.info2('o_a:', self.o_a)
        #ut.info2('ro_w:\n', self.ro_w)
        
