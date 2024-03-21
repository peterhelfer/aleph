#!/usr/bin/env python
#
"""
make_dataset.py
~~~~~~
Generate a synthetic dataset for training and testing the LSM
By Peter Helfer, 2019
"""

#### Libraries
# Standard library imports
import os, sys, getopt, pickle, gzip, enum

# Third-party library imports
import numpy as np

# Local library imports
#
scriptpath = "../lib/"
sys.path.append(os.path.abspath(scriptpath))

import util as ut

# Project imports
#
import Dataset as ds

# Definitions

# pattern     = list of firing rates
# pattern_set = list of patterns, one per channel
# context     = an int, indicates super-category
# label       = an int, indicates category
# labeled_pattern_set = a tuple of a context, a label and a pattern_set

# block   = list of ints representing time frames with (1) or without (0) spikes,
#           generated with some specified average firing rate
# stream  = concatenation of blocks with specified spiking rates,
#           all with the same specified block length and time frame length
# sample  = a list of streams, one per channel
# labeled_sample = a tuple of a context, a label and a sample
# dataset = a list of labeled_samples

# Parameters for generating the dataset, with default values
# (Some non-defaults can be specified on the command line)
#
frame_length        =   1    # (msec) length of time frame (inverse of clock rate)
min_rate            =   0    # minimum firing rate
max_rate            = 200    # maximum firing rate

def visualize_stream(stream, block_size):
    """Map a stream (list of 0s and 1s) to a string of '-' and 'X'
    and insert a '|' at every block_size elements"""
    s = ""
    for i, x in enumerate(stream):
        if i % block_size == 0:
            s += '|'
        s += 'X' if x else '-'
    s += '|'
    return s

def visualize_labeled_samples(labeled_samples, block_size):
    """Print an ASCII representation of the dataset on stdout"""
    print('num_samples: {}'.format(len(labeled_samples)))
    for i, ls in enumerate(labeled_samples):
        print("Sample {} context={} label={}:".format(i, ls.context, ls.label))
        for stream in ls.sample:
            print(visualize_stream(stream, block_size))

# Specify pattern_id to label mappings for each context
#
context_specs = []
for values in [
        # ctx, pattern_id, label
        [0, 0, 0],
        [0, 1, 0],
        [0, 2, 1],
        [0, 3, 1],
        [1, 0, 0],
        [1, 1, 0],
        [1, 2, 1],
        [1, 3, 0]]:
    context_specs.append(ds.ContextSpec(*values))

def main():
    # Process command line
    #
    parser = ut.TerseArgumentParser()
    parser.add_bool_arg('visualize', 'Visualize the dataset on stdout')
    parser.add_str_arg('generator_type', 'poisson', 'Generator type: poisson, fixed_rate, deterministic, or rate_based')
    parser.add_bool_arg('rate_based', 'no spikes, inject rate as current')
    parser.add_bool_arg('debug_mode', 'Debug mode')
    parser.add_int_arg('num_channels', 5, 'number of channels')
    parser.add_int_arg('num_samples_per_pattern', 10, 'number of samples per pattern')
    parser.add_int_arg('block_size', 40, 'block length (msec)')
    parser.add_int_arg('num_blocks', 5, 'blocks per sample')
    parser.add_argument('ds_fname', metavar='DS_FILE', type=str, help='Dataset file')

    global args
    args = parser.parse_args()

    generator_type = ds.Generator.from_str(args.generator_type)

    if args.debug_mode:
        # To facilitate debugging
        np.random.seed(0) 

    # Generate the dataset
    #
    d = ds.make_random_dataset(context_specs, args.num_samples_per_pattern,
                               args.block_size, frame_length,
                               args.num_channels, args.num_blocks,
                               min_rate, max_rate,
                               generator_type=generator_type)

    if args.visualize:
        print(d.description)
        visualize_labeled_samples(d.labeled_samples, args.block_size)
        
    # Write it to file
    #
    f = gzip.open(args.ds_fname, 'wb')
    pickle.dump(d, f)
    f.close()

if __name__ == "__main__":
    main()
