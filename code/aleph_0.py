#!/usr/bin/env python
#
"""
aleph_0.py
~~~~~~

The aleph_0 main program, driver for the aleph_0_net network.

By Peter Helfer, 2023
"""

# Standard library imports
#
import os, sys, gzip, pickle, math, time, enum

# disable CPU affinity so that numpy can multi-core
# (maybe not needed)
#os.system("taskset -p 0xff %d" % os.getpid())

# Make sure OpenBLAS is configured to multi-thread
#
os.environ['OPENBLAS_NUM_THREADS'] = '8'
#os.environ["MKL_NUM_THREADS"] = "100"
#os.environ["NUMEXPR_NUM_THREADS"] = "100"
#os.environ["OMP_NUM_THREADS"] = "100"

# Don't use GPU (not sure if this works)
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Third-party library imports
#
import numpy as np
from line_profiler import LineProfiler

# Local library imports
#
import util as ut
import Dataset
import aleph_net

# Instantiate a random number generator
#
rng = np.random.default_rng()

# Hyper-parameters can be specified on the command line;
# default values are given below.
#
args = {} # Command-line arguments

def parse_cmdline():
    # Get aleph_net's argument parser
    #
    parser = aleph_net.get_arg_parser()

    # Add our arguments to the parser
    #

    # Network topology
    #
    parser.add_int_arg('num_subnets',           3,      'Number of subnets')
    parser.add_int_arg('subnet_size',           100,    'Number of neurons per subnets')

    parser.add_str_arg('trace_file',            None, 'Trace file. {} will be replaced by pid')
    parser.add_bool_arg('debug_rand',                 'Force same random sequence every run')
    parser.add_bool_arg2('d', 'debug_warn',           'Treat warnings as errors')
    parser.add_nonneg_int_arg('num_train_samples',
                                                None, 'Number of train samples. None means 90%% of the dataset.')
    parser.add_nonneg_int_arg('num_test_samples',
                                                None, 'Number of test samples. None means 10%% of the dataset.')
    parser.add_bool_arg('confusion_matrix',           'Display confusion matrix')

    # Input file (required parameter)
    #
    parser.add_argument('ds_fname', metavar='DS_FILE', type=str, help='Dataset file name')

    # Control recording of neuron states info for post-mortem analysis
    #
    parser.add_bool_arg('rec_train',                     'Record during training')
    parser.add_bool_arg('rec_test',                      'Record during testing')
    parser.add_int_arg('rec_epoch',             0,       'Training epoch during which to record')
    parser.add_int_arg('rec_start_sample',      0,       'Train/test sample during which to start recording')
    parser.add_int_arg('rec_num_samples',       1,       'Number of samples during which to record')
    parser.add_str_arg('rec_filename',          None,    'Recording file name. None means generate unique name')

    # Parse the command line
    #
    global args
    args = parser.parse_args()

    # Handle trace_file parameter
    #
    if args.trace_file is not None:
        trace_file = args.trace_file.format(os.getpid())
        file = open(trace_file, 'a')
        ut.set_trace_file(file)

    # Handle debug_warn parameter
    #
    if args.debug_warn:
        np.seterr(all='raise') # Treat floating point warnings as errors

is_recording = False

def start_recording(nw):
    """Start recording nw's neuron states"""
    global is_recording
    nw.init_recording()
    is_recording = True

def stop_recording(nw):
    """Stop recording nw's neuron states and save the recording to file"""
    global is_recording
    recording = nw.get_recording()
    nw.stop_recording()
    is_recording = False
    
    # Construct a unique filename
    #
    rec_filename = args.rec_filename
    if rec_filename is None:
        rec_filename = 'aleph_state'
    rec_filename = f'{rec_filename}_{ut.now_str()}'
    
    with open(rec_filename, 'wb') as rec_file:
        pickle.dump(recording, rec_file)

def load_data(fname):
    """Read the data from file and return it."""
    try:
        try:
            f = gzip.open(fname, 'rb')
            data = pickle.load(f)
        except OSError as exc:
            f = open(fname, 'rb')  # Allow gzipped or plain pickle files.
            data = pickle.load(f)

        f.close()
        return data
    except IOError as exc:
        ut.trace(ut.Trace.FATAL, exc)

# Trace message printer with min interval between traces
#
class TraceMsg:
    def __init__(self):
        self.last_trace_time = 0
    
    def print(self, str, min_intrvl):
        """Print a trace message if min_intrvl has elapsed 
        since last time"""
        now = time.time()
        if now - self.last_trace_time >= min_intrvl:
            ut.info2(str)
            self.last_trace_time = now

trace_msg = TraceMsg()

#@profile
def train(nw, train_data, num_epochs):
    """Train the NW to map samples to labels
    :param nw: Main network.
    :train_data: Training data
    :num_epochs: number of training epochs
    """
    # Print max 100 trace messages per epoch and no more often than
    # every min_trace_intrvl seconds
    #
    num_train_samples = len(train_data)
    trace_sample_intrvl = pow(10, math.ceil(math.log10(num_train_samples/100)))
    min_trace_intrvl = 10
    ut.info2('Training')

    # Train the NW (unless it is pre-trained)
    #
    for e in range(num_epochs):
        ut.info2(f'Epoch: {e + 1}/{num_epochs}')

    for i, labeled_sample in enumerate(train_data):
        # Maybe start recording
        #
        if args.rec_train and e == args.rec_epoch and i == args.rec_start_sample:
            start_recording(nw)
                
        # Maybe stop recording
        #
        if is_recording and (i - args.rec_start_sample >= args.rec_num_samples):
            stop_recording(nw)
        
        # Process one sample
        #
        sample = np.array(labeled_sample.sample).T
        label = labeled_sample.label

        if i % trace_sample_intrvl == 0:
            trace_msg.print(f'Training NW epoch {e+1}/{args.num_epochs} ' +
                            f'sample {i}/{len(train_data)} ',
                                min_trace_intrvl)

                
def test(nw, test_data):
    """For each test sample, use NW to infer the label.
    :param nw: Main network
    :param test_data: Set of samples to test
    :return: Fraction of correct inferences
    """

    num_correct_label = 0

    if args.confusion_matrix:
        conf_mat = ut.ConfusionMatrix(
            nw.num_outputs, title=str(args.ds_fname) + ' (%)', # str() handles None case
            percent='col',
            xlabel='Target', ylabel='Inferred')

    n = len(test_data)
    for i, labeled_sample in enumerate(test_data):
        # Maybe start recording
        #
        if args.rec_test  and i == args.rec_start_sample:
            start_recording(nw)
                
        # Maybe stop recording
        #
        if is_recording and (i - args.rec_start_samples >= args.rec_num_samples):
            stop_recording(nw)

        label = labeled_sample.label

        sample = np.array(labeled_sample.sample).T
        true_label = labeled_sample.label
        prediction = nw.infer(sample, None, true_label)
        num_correct_label += (prediction == true_label)
        #print(f'INFER: {label} --> {prediction}')
        if args.confusion_matrix:
            conf_mat.add_value(prediction, true_label)
        prev_prediction = prediction
        
    score = num_correct_label / n
            
    ut.info(f'label score: {num_correct_label}/{n} = {score}')

    if args.confusion_matrix:
        conf_mat.plot()

    return score

def main():
    global rng, readout_map

    # Process command line
    #
    parse_cmdline()

    ut.set_trace_level_str(args.trace_level)

    ut.info2("===== cmdline parameters =====", notime=True)
    ut.info2(ut.dict_to_str(vars(args)), notime=True)
    ut.info2("==============================", notime=True)

    if args.debug_rand:
        # Force same random sequence every run
        #
        rng = np.random.default_rng(seed=0)

    # Load the dataset
    #
    data = load_data(args.ds_fname)
   
    if isinstance(data, Dataset.Dataset):
        # Classic dataset: single list of samples.  A Dataset is a list of
        # labeled samples. Each sample is a tuple of (id, sample), where id
        # is an int and sample is a list of streams, one stream per input
        # neuron. A stream is a sequence of ones and zeros indicating
        # whether the input neuron should or should not be active in the
        # corresponding time frame.

        ds = data
        
        # Split the dataset into train and test data
        #
        num_samples = len(ds.labeled_samples)
    
        if args.num_train_samples in (None, 0):
            if args.num_test_samples in (None, 0):
                # By default, use 10% of the samples for testing
                args.num_test_samples = int(num_samples / 10)
            args.num_train_samples = num_samples - args.num_test_samples
        else:
            if args.num_test_samples in (None, 0):
                args.num_test_samples = min(num_samples - args.num_train_samples,
                                            int(args.num_train_samples / 9))
        ut.info(f'num_train: {args.num_train_samples}  ' +
                f'num_test: {args.num_test_samples}')
    
        if (args.num_train_samples <=0
            or args.num_test_samples <= 0
            or args.num_train_samples + args.num_test_samples > num_samples):
            ut.fatal("The specified num_train_samples and " +
                     "num_test_samples are incompatible with the " +
                     "number of samples in the data file") 
            
        # Shuffle the samples (unless in PREVLBL mode) and take train and test
        # samples from either end
        #
    
        train_data = ds.labeled_samples[:args.num_train_samples]
        test_data = ds.labeled_samples[-args.num_test_samples:]

    elif ut.is_listof(data, Dataset.Task):
        tasks = data # just to be clear
        ut.abort_if(len(tasks) != 1, 'Multiple tasks not supported for now')

        # A Task contains train and test data, no need to split them.
        #
        train_data = tasks[0].train_dataset.labeled_samples[:args.num_train_samples]
        test_data = tasks[0].test_dataset.labeled_samples[:args.num_test_samples]


    else:
        ut.abort('Unknown data file type.')

    # Set num_inputs to the number of streams in a sample
    #
    num_inputs = len(train_data[0].sample)

    # Get ordered list of labels that occur in the dataset
    #
    labels = np.unique([ls.label for ls in train_data] + [ls.label for ls in test_data])

    # Set num_outputs to the number of unique labels
    #
    num_outputs = len(labels)
        

    # Build the network
    #
    nw = aleph_net.Network('NW', num_inputs, args.num_subnets, args.subnet_size, rng, args)

    nw.dump_topology()
    #nw.dump_connectivity()

    # Train the network
    #
    train(nw, train_data, args.num_epochs)
    
    # Test it
    #
    acc = test(nw, test_data)
    ut.info(f'accuracy = {acc:.03f}')

    # Print accuracy as a single float number, as required by ga.py
    #
    print(f'{acc:.03f}')

if __name__ == '__main__':

    PROFILER=''

    if PROFILER == 'LP':
        lp = LineProfiler()
        lp_wrapper = lp(main)
        lp_wrapper()
        #lp.print_stats()
    elif PROFILER == 'CP':
        stats_file = f'stats_{os.getpid()}'
        import cProfile
        import re
        cProfile.run('main()', stats_file)

        import pstats
        from pstats import SortKey
        p = pstats.Stats(stats_file)
        p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)
    else:
        main()
