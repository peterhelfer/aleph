#!/usr/bin/env python
#
"""
view_dataset.py
~~~~~~
Print an ASCII representation of an LSM test dataset to stdout
By Peter Helfer
Created: 2019
Last mofified: 2021-03-14
"""

# Standard library imports
#
import os, sys, getopt, pickle, gzip, enum
import numpy as np

# Local library imports
#
scriptpath = "../lib/"
sys.path.append(os.path.abspath(scriptpath))

import util as ut

# Process command line arguments
#
params = {}

def parse_cmdline():
    parser = ut.TerseArgumentParser()
    parser.add_str_arg2('t', 'trace_level', 'INFO', 'trace level')
    parser.add_bool_arg('no_description', "Don't print out the description")
    parser.add_bool_arg('no_samples', "Don't print out the samples")
    parser.add_bool_arg('no_separation', "Don't print out the separation property")
    parser.add_argument('ds_fname', metavar='DATASET_FILENAME', type=str, help='Dataset file name')

    global params
    params   = parser.parse_args()

#Globals
#
block_size = 20

def load_dataset(fname):
    """Read the dataset from file and return it as a list of labeled
    samples. Each sample is a tuple of (id, sample), where id is an int and
    sample is a list of streams, one stream per input neuron. A stream is a
    sequence of ones and zeros indicating whether the input neuron should or
    should not be active in the corresponding time frame."""
    try:
        try:
            f = gzip.open(fname, 'rb')
            ds = pickle.load(f)
        except OSError as exc:
            f = open(fname, 'rb')  # Allow gzipped or plain pickle files.
            ds = pickle.load(f)

        f.close()
        return ds
    except IOError as exc:
        ut.trace(ut.Trace.FATAL, exc)

def visualize_stream(stream, block_size):
    """Map a stream (list of 0s and 1s) to a string of '-' and 'X'
    and insert a '|' at every block_size element"""
    s = ""
    for i, x in enumerate(stream):
        if i % block_size == 0:
            s += '|'
        s += 'X' if x else '-'
    s += '|'
    return s

def visualize_dataset(ds, block_size):
    """Print an ASCII representation of the dataset on stdout"""
    print('num_samples: {}'.format(len(ds.labeled_samples)))
    for i, ls in enumerate(ds.labeled_samples):
        print("Sample {} context={} label={}:".format(i, ls.context, ls.label))
        for stream in ls.sample:
            print(visualize_stream(stream, block_size))

progname=''

def usage():
    print('Usage: ' + progname + ' [-h]' + ' input_file_name')
    print('  -h: Print this message')
    sys.exit(2)

def main():
    progname = os.path.basename(sys.argv[0])

    # Process command line
    #
    parse_cmdline()

    # load the dataset
    #
    ds = load_dataset(params.ds_fname)

    # show it
    #
    global block_size
    if not params.no_description:
        print(ds.description)

    if not params.no_samples:
        try:
            block_size = ds.block_size
        except AttributeError as exc:
            try:
                block_size = ds.block_length # old name
            except:
                print(exc)
                ut.info(f'old .ds file, using default block_size: {block_size}')
    
        visualize_dataset(ds, block_size)

    if not params.no_separation:
        if 0:
            separation, mean_sq_inter_class_distance, mean_intra_class_variance = ds.get_pattern_separation()

            print('Pattern Separation:')
            print(f'    mean_sq_inter_class_distance = {mean_sq_inter_class_distance:.02f}')
            print(f'    mean_intra_class_variance    = {mean_intra_class_variance:.02f}')
            print(f'    separation = {separation:.02f}')

        if 1:
            separation, mean_sq_inter_class_distance, mean_intra_class_variance = ds.get_sample_separation()

            print('Sample Separation:')
            print(f'    mean_sq_inter_class_distance = {mean_sq_inter_class_distance:.02f}')
            print(f'    mean_intra_class_variance    = {mean_intra_class_variance:.02f}')
            print(f'    separation = {separation:.02f}')

if __name__ == "__main__":
    main()
        
