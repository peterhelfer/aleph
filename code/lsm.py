#!/usr/bin/env python
#
"""
lsm.py
~~~~~~

An implementation of a liquid state machine.

By Peter Helfer, 2019
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
import lsm_net

# Instantiate a random number generator
#
rng = np.random.default_rng()

# Context mode
#
class CtxMode(ut.StrEnum):
    NONE = 'none'
    RANDOM = 'random' # Randomly asssign labels to num_ctx contexts
    EXPLICIT = 'explicit'  # Context ID is specified for each sample in the dataset
    PREVLBL = 'prevlbl' # Previous sample's label is used as context

ctx_mode = CtxMode.NONE

# LSM save/load mode
#
class LsmFileMode(ut.StrEnum):
    TRAINED = 'trained'      # Save LSM after training and don't train after loading
    UNTRAINED = 'untrained'  # Save LSM before training and train after loading

cid_file_mode = None
mnw_file_mode = None

# Hyper-parameters can be specified on the command line;
# default values are given below.
#
args = {} # Command-line arguments

def parse_cmdline():
    # Get lsm_net's argument parser
    #
    parser = lsm_net.get_arg_parser()

    # Add our arguments to the parser
    #
    parser.add_str_arg('trace_file',            None, 'Trace file. {} will be replaced by pid')
    parser.add_bool_arg('debug_rand',                 'Force same random sequence every run')
    parser.add_bool_arg2('d', 'debug_warn',           'Treat warnings as errors')
    parser.add_nonneg_int_arg('num_train_samples',
                                                None, 'Number of train samples. None means 90%% of the dataset.')
    parser.add_nonneg_int_arg('num_test_samples',
                                                None, 'Number of test samples. None means 10%% of the dataset.')
    parser.add_str_arg('ctx_mode',              'none', 'Context mode: None, random, prevlbl or explicit')
    parser.add_nonneg_int_arg('num_ctx',        0,    'Number of contexts (used when ctx_mode=random)')
    parser.add_bool_arg('ctx_oracle',                 'Use true context for inference (default: infer from sample)')
    parser.add_float_arg('ora_acc',             1.0,  'Context oracle accuracy')
    parser.add_bool_arg('shared_readout',             'Share readout nodes between contexts')
    parser.add_bool_arg('confusion_matrix',           'Display confusion matrix')

    parser.add_bool_arg('print_separation',           'Print separation (in addition to accuracy)')
    parser.add_str_arg('mnw_save_file',         None, 'Save the main network to this file')
    parser.add_str_arg('cid_save_file',         None, 'Save the context identifier network to this file')
    parser.add_str_arg('mnw_load_file',         None, 'Load the main network from this file')
    parser.add_str_arg('cid_load_file',         None, 'Load the context identifier network from this file')
    parser.add_str_arg('cid_file_mode',    'trained', 'Save or load CID "trained" or "untrained"')
    parser.add_str_arg('mnw_file_mode',    'trained', 'Save or load MNW "trained" or "untrained"')
    
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

    # Handle ctx_mode parameter
    #
    global ctx_mode
    ctx_mode = CtxMode.from_str(args.ctx_mode)

    # Handle cid_file_mode and mnw_file_mode parameters
    #
    global cid_file_mode, mnw_file_mode
    cid_file_mode = LsmFileMode.from_str(args.cid_file_mode)
    mnw_file_mode = LsmFileMode.from_str(args.mnw_file_mode)
    
    # Sanity checks
    #
    ut.abort_if(args.shared_readout and (ctx_mode != CtxMode.RANDOM),
                '--shared_readout is only supported with --ctx_mode RANDOM')

    ut.abort_if((args.cid_load_file is not None) and (ctx_mode is None),
                '--cid_load_file with --ctx_mode None makes no sense')

# Save/load an LSM instance
#
def save_lsm(lsm, filename):
    """Save the lsm to file.
    :param lsm: The LSM
    :param filename: file name to save to
    """
    with open(filename, 'wb') as lsm_file:
        pickle.dump(lsm, lsm_file)

def load_lsm(filename, name, num_inputs, num_outputs):
    """Load the lsm from file.
    :param filename: file name to load from
    :param name: set LSM name to this
    :param num_inputs, num_outputs: expected 
    :return: The LSM
    """
    with open(filename, 'rb') as lsm_file:
        lsm = pickle.load(lsm_file)

    if (num_inputs is not None) and (lsm.num_inputs != num_inputs):
        ut.fatal('lsm.num_inputs != num_inputs')

    if (num_outputs is not None) and (lsm.num_outputs != num_outputs):
        ut.fatal('lsm.num_outputs != num_outputs')

    return lsm

num_ctx = 0 # Will be set in main()

is_recording = False

def start_recording(mnw):
    """Start recording mnw's neuron states"""
    global is_recording
    mnw.init_recording()
    is_recording = True

def stop_recording(mnw):
    """Stop recording mnw's neuron states and save the recording to file"""
    global is_recording
    recording = mnw.get_recording()
    mnw.stop_recording()
    is_recording = False
    
    # Construct a unique filename
    #
    rec_filename = args.rec_filename
    if rec_filename is None:
        rec_filename = 'lsm_state'
    rec_filename = f'{rec_filename}_{ut.now_str()}'
    
    with open(rec_filename, 'wb') as rec_file:
        pickle.dump(recording, rec_file)
        

# Label-to-context dictionary (used in CtxMode.RANDOM)
#
class LabelCtxDict:
    def __init__(self, lbls):
        if num_ctx > 0:
            self.dict = {}
            ctx = 0
            for lbl in lbls:
                self.dict[lbl] = ctx
                ctx = (ctx + 1) % num_ctx

    def get_ctx(self, lbl):
        """Get context for lbl"""
        if num_ctx > 0:
            return self.dict[lbl]
        else:
            return None

    def get_lbls(self, ctx):
        """Get labels that are mapped to context ctx"""
        return [lbl for (lbl, c) in self.dict.items() if c == ctx]

lbl_ctx_dict = None # Will be initialized in main()

def ctx_oracle(true_ctx):
    """Return true_ctx with probability args.ora_acc"""
    if num_ctx == 1 or rng.random() < args.ora_acc:
        ctx = true_ctx
    else:
        # pick any of the others with equal prob
        ctx = rng.choice(np.delete(range(num_ctx), true_ctx))
    return ctx

# Purposes that context IDs may be used for
#
class CtxPurpose(ut.StrEnum):
    ACTUAL_CTX = 'actual_ctx' # Obtain the actual context for a sample
    TRAIN_MNW  = 'train_mnw'  # To modulate the MNW during training
    TEST_MNW   = 'test_mnw'   # To modulate the MNW during testing

def select_ctx(*, purpose, ctx_mode, labeled_sample, prevlbl=None,
               cid=None, actual_ctx=None):
    """Select context to use with a sample
    :param purpose: What the context will be used for
    :param ctx_mode: Context mode
    :param labeled_sample: A labeled sample
    :param prevlbl: previously trained or inferred label
    :param cid: context identifier network
    :param actual_ctx: used to create CID xtraces
    """
    # How to determine the ctx_id:
    #
    class CtxSrc(ut.StrEnum):
        DICT     = 'dict'     # Use the label for lookup in lbl_ctx_dict
        DICT_ORA = 'dict_ora' # dictionary-based oracle
        SAMPLE   = 'sample'   # use labeled_sample's ctx field
        SMPL_ORA = 'smpl_ora' # sample-based oracle
        PREVLBL  = 'prevlbl'  # use prevlbl
        INFER    = 'infer'    # use CID to infer ctx
        NONE     = 'none'     # None

    if purpose == CtxPurpose.ACTUAL_CTX:
        if   ctx_mode == CtxMode.RANDOM:     src = CtxSrc.DICT
        elif ctx_mode == CtxMode.EXPLICIT:   src = CtxSrc.SAMPLE
        elif ctx_mode == CtxMode.PREVLBL:    src = CtxSrc.PREVLBL
        else: ut.fatal(f'Unknown CtxMode: {ctx_mode}')

    elif purpose == CtxPurpose.TRAIN_MNW:
        if   ctx_mode == CtxMode.RANDOM:     src = CtxSrc.DICT  # Later: CID
        elif ctx_mode == CtxMode.EXPLICIT:   src = CtxSrc.SAMPLE
        elif ctx_mode == CtxMode.PREVLBL:    src = CtxSrc.PREVLBL
        else: ut.fatal(f'Unknown CtxMode: {ctx_mode}')

    elif purpose == CtxPurpose.TEST_MNW:
        if   ctx_mode == CtxMode.RANDOM:
            if args.ctx_oracle:              src = CtxSrc.DICT_ORA
            else:                            src = CtxSrc.INFER
        elif ctx_mode == CtxMode.EXPLICIT:
            if args.ctx_oracle:              src = CtxSrc.SMPL_ORA
            else:                            src = CtxSrc.INFER
        elif ctx_mode == CtxMode.PREVLBL:    src = CtxSrc.PREVLBL
        else: ut.fatal(f'Unknown CtxMode: {ctx_mode}')

    else:
        ut.fatal(f'Unknown CtxPurpose: {purpose}')

    if   src == CtxSrc.DICT:     ctx = lbl_ctx_dict.get_ctx(labeled_sample.label)
    elif src == CtxSrc.DICT_ORA: ctx = ctx_oracle(lbl_ctx_dict.get_ctx(labeled_sample.label))
    elif src == CtxSrc.SAMPLE:   ctx = labeled_sample.context
    elif src == CtxSrc.SMPL_ORA: ctx = ctx_oracle(labeled_sample.context)
    elif src == CtxSrc.PREVLBL:  ctx = prevlbl
    elif src == CtxSrc.INFER:    ctx = cid.infer(np.array(labeled_sample.sample).T, None, actual_ctx)
    elif src == CtxSrc.NONE:     ctx = None
    else: ut.fatal(f'Unknown CtxSrc: {src}')

    return ctx


# LabelSqueezer
#   The set of labels in the dataset may be something like [2, 4, 6, 8],
#   i.e. not consecutive and/or not zero-based.
#   This dictionary maps them to/from a compact zero-based set like
#   [0, 1, 2, ...], making it easy to map labels to the LSM's readout
#   neurons.
#
class LabelSqueezer:
    def __init__(self, lbls):
        self.dict = {}
        next_sqz_lbl = 0
        for lbl in lbls:
            self.dict[lbl] = next_sqz_lbl
            next_sqz_lbl += 1

    def squeeze(self, lbl):
        """Get sqz_lbl for lbl"""
        return self.dict[lbl]

    def unsqueeze(self, sqz_lbl):
        """Get lbl for sqz_lbl"""
        return [k for k, v in self.dict.items() if v == sqz_lbl][0]

lbl_squeezer = None # Will be initialized in main()


# ReadoutMap
#   Map labels to readout values:
#   In a shared-readout setup, the n labels assigned to each context are
#   mapped to the same n readout values. Example: contexts a and b may
#   include samples with, say, labels {2,1,5} and {0,4,3}, respectively; the
#   network will be trained to classify each context using readout values
#   {0,1,3}. So labels 2 and 0 will both be mapped to 0, etc.
#
class ReadoutMap:
    def __init__(self):
        self.dict = {}
        if num_ctx > 1 and args.shared_readout:
            for ctx in range(num_ctx):
                val = 0
                lbls = lbl_ctx_dict.get_lbls(ctx)
                for lbl in lbls:
                    self.dict[lbl] = val
                    val += 1

    def get_val(self, lbl):
        if num_ctx > 1 and args.shared_readout:
            return self.dict[lbl]
        else:
            return lbl

readout_map = None # Will be initialized in main()

def readout_target(label):
    """Apply mappings to obtain target readout value for a label.
    :param label: A label
    :return: The corresponding readout value
    """
    return readout_map.get_val(lbl_squeezer.squeeze(label))

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
def train(mnw, cid, train_data, num_epochs):
    """Train the CID to map samples to context IDs, use the context IDs 
    to modulate the MNW, and train the MNW to map samples to labels
    :param mnw: Main network.
    :param cid: Context identifier network.
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

    labeled_cid_xtrace_arrays = None
    labeled_mnw_xtrace_arrays = None

    # If we have a CID and it is not pre-trained, train it
    #
    if (cid is not None) and \
       ((args.cid_load_file is None) or (cid_file_mode == LsmFileMode.UNTRAINED)):
        for e in range(num_epochs):
            ut.info2(f'Epoch: {e + 1}/{num_epochs}')
    
            if e > 0:
                # Since the reservoir is fixed, the same input will produce the
                # same xtrace every time. To save time, we reuse the xtraces
                # from epoch 0.
                #
                if labeled_cid_xtrace_arrays is None:
                    labeled_cid_xtrace_arrays = cid.get_xtraces_acc()
    
            cid.clear_xtraces_acc()
            if labeled_cid_xtrace_arrays is not None:
                rng.shuffle(labeled_cid_xtrace_arrays)
                for i, labeled_xtraces in enumerate(labeled_cid_xtrace_arrays):
                    # Process one set of xtraces
                    #
                    ctx = labeled_xtraces[0]
                    xtraces = labeled_xtraces[1]
                    cid.train(None, xtraces, ctx, ctx)
            else:
                for i, labeled_sample in enumerate(train_data):
                    # Process one sample
                    #
                    sample = np.array(labeled_sample.sample).T
    
                    ctx = select_ctx(purpose=CtxPurpose.ACTUAL_CTX,
                                     ctx_mode=ctx_mode,
                                     labeled_sample=labeled_sample)
                    
                    if i % trace_sample_intrvl == 0:
                        trace_msg.print(f'Training CID epoch {e+1}/{args.num_epochs} ' +
                                f'sample {i}/{len(train_data)}',
                                        min_trace_intrvl)
        
                    cid.train(sample, None, ctx, ctx)


    # Train the MNW (unless it is pre-trained)
    #
    if (args.mnw_load_file is None) or (mnw_file_mode == LsmFileMode.UNTRAINED):
        for e in range(num_epochs):
            ut.info2(f'Epoch: {e + 1}/{num_epochs}')
    
            if e > 0:
                # Since the reservoir is fixed, the same input will produce the
                # same xtrace every time. To save time, we reuse the xtraces
                # from epoch 0.
                #
                if labeled_mnw_xtrace_arrays is None:
                    labeled_mnw_xtrace_arrays = mnw.get_xtraces_acc()
    
            mnw.clear_xtraces_acc()
            
            if labeled_mnw_xtrace_arrays is not None:
                if ctx_mode != CtxMode.PREVLBL:
                    rng.shuffle(labeled_mnw_xtrace_arrays)
    
                for i, labeled_xtraces in enumerate(labeled_mnw_xtrace_arrays):
                    # Process one set of xtraces
                    #
                    label = labeled_xtraces[0]
                    xtraces = labeled_xtraces[1]
                    mnw.train(None, xtraces, readout_target(label), label)
            else:
                prevlbl = num_ctx - 1 # Used as initial context for PREVLBL mode
    
                for i, labeled_sample in enumerate(train_data):
                    # Maybe start recording
                    #
                    if args.rec_train and e == args.rec_epoch and i == args.rec_start_sample:
                        start_recording(mnw)
                            
                    # Maybe stop recording
                    #
                    if is_recording and (i - args.rec_start_sample >= args.rec_num_samples):
                        stop_recording(mnw)
                    
                    # Process one sample
                    #
                    sample = np.array(labeled_sample.sample).T
                    label = labeled_sample.label
    
                    ctx = None
                    if ctx_mode != CtxMode.NONE:
                        ctx = select_ctx(purpose=CtxPurpose.TRAIN_MNW,
                                         ctx_mode=ctx_mode,
                                         labeled_sample=labeled_sample,
                                         prevlbl=prevlbl)
    
                    if i % trace_sample_intrvl == 0:
                        trace_msg.print(f'Training MNW epoch {e+1}/{args.num_epochs} ' +
                                        f'sample {i}/{len(train_data)} ',
                                        min_trace_intrvl)
        
                    if ctx is not None:
                        mnw.modulate(ctx)
    
                    mnw.train(sample, None, readout_target(label), label)
                    prevlbl = label
                
def test(mnw, cid, test_data):
    """For each test sample, use CID to infer ctx, modulate MNW with ctx, 
    and use MNW to infer the label.
    :param mnw: Main network
    :param cid: Context identification network
    :param test_data: Set of samples to test
    :return: Fraction of correct inferences
    """

    num_correct_label = 0
    num_correct_ctx = 0
    prev_prediction = num_ctx - 1 # Used as initial context in PREVLBL mode

    if args.confusion_matrix:
        conf_mat = ut.ConfusionMatrix(
            mnw.num_outputs, title=str(args.ds_fname) + ' (%)', # str() handles None case
            percent='col',
            xlabel='Target', ylabel='Inferred')

    def set_up_ctx_modulation(labeled_sample):
        """Set up context modulation"""

        # Determine actual_ctx for diagnostic purposes
        #
        actual_ctx = select_ctx(purpose=CtxPurpose.ACTUAL_CTX,
                         ctx_mode=ctx_mode,
                         labeled_sample=labeled_sample,
                         prevlbl=prev_prediction)

        # Determine context for modulation of the MNW
        #
        ctx = select_ctx(purpose=CtxPurpose.TEST_MNW,
                         ctx_mode=ctx_mode,
                         labeled_sample=labeled_sample,
                         prevlbl=prev_prediction,
                         cid=cid,
                         actual_ctx=actual_ctx)

        nonlocal num_correct_ctx
        ut.info3(f'actual_ctx = {actual_ctx}  ctx = {ctx}')
        num_correct_ctx += (ctx == actual_ctx)
        mnw.modulate(ctx)

    n = len(test_data)
    for i, labeled_sample in enumerate(test_data):
        # Maybe start recording
        #
        if args.rec_test  and i == args.rec_start_sample:
            start_recording(mnw)
                
        # Maybe stop recording
        #
        if is_recording and (i - args.rec_start_samples >= args.rec_num_samples):
            stop_recording(mnw)

        label = labeled_sample.label

        if ctx_mode != CtxMode.NONE:
            set_up_ctx_modulation(labeled_sample)

        sample = np.array(labeled_sample.sample).T
        true_label = labeled_sample.label
        prediction = mnw.infer(sample, None, true_label)
        num_correct_label += (prediction == readout_target(true_label))
        #print(f'INFER: {label} --> {prediction}')
        if args.confusion_matrix:
            conf_mat.add_value(prediction, true_label)
        prev_prediction = prediction
        
    score = num_correct_label / n
            
    ut.info(f'ctx score: {num_correct_ctx}/{n} = ' +
            f'{num_correct_ctx / n}')
    ut.info(f'label score: {num_correct_label}/{n} = {score}')

    if args.confusion_matrix:
        conf_mat.plot()

    return score

def main():
    global rng, ctx_mode, num_ctx, lbl_ctx_dict, lbl_squeezer, readout_map

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
    
        if ctx_mode != CtxMode.PREVLBL:
            rng.shuffle(ds.labeled_samples)
    
        train_data = ds.labeled_samples[:args.num_train_samples]
        test_data = ds.labeled_samples[-args.num_test_samples:]

    elif ut.is_listof(data, Dataset.Task):
        tasks = data # just to be clear
        ut.abort_if(len(tasks) != 1, 'Multiple tasks not supported for now')

        # A Task contains train and test data, no need to split them.
        #
        train_data = tasks[0].train_dataset.labeled_samples[:args.num_train_samples]
        test_data = tasks[0].test_dataset.labeled_samples[:args.num_test_samples]

        if ctx_mode != CtxMode.PREVLBL:
            rng.shuffle(train_data)
            rng.shuffle(test_data)

    else:
        ut.abort('Unknown data file type.')

    # Set num_inputs to the number of streams in a sample
    #
    num_inputs = len(train_data[0].sample)

    # Get ordered list of labels that occur in the dataset
    #
    labels = np.unique([ls.label for ls in train_data] + [ls.label for ls in test_data])

    # Set up context stuff
    #
    if ctx_mode == CtxMode.RANDOM:
        # Randomly assign labels to contexts
        #
        num_ctx = args.num_ctx
        rlabels = labels.copy()
        rng.shuffle(rlabels)
        lbl_ctx_dict = LabelCtxDict(rlabels)
    elif ctx_mode == CtxMode.EXPLICIT:
        # Count the number of ctx IDs in the dataset
        num_ctx = max([ls.context for ls in train_data + test_data]) + 1
    elif ctx_mode == CtxMode.PREVLBL:
        # Count the number of labels in the dataset + 1 for initial ctx
        num_ctx = max([ls.label for ls in train_data +  test_data]) + 1
    else:
        num_ctx = 0

    # Initialize lbl_squeezer
    #
    lbl_squeezer = LabelSqueezer(labels)

    # Initialize ReadoutMap
    #
    readout_map = ReadoutMap()

    # Set num_outputs to the number of unique readout values
    #
    squeezed_lbls = [lbl_squeezer.squeeze(lbl) for lbl in labels]
    readout_vals = np.unique([readout_map.get_val(sqlab) for sqlab in squeezed_lbls])
    num_outputs = len(readout_vals)

        

    # Build or load the main network
    #
    if args.mnw_load_file is not None:
        mnw = load_lsm(args.mnw_load_file, 'MNW', num_inputs, num_outputs)
        mnw.stop_recording() # This ensures backwards compatibility with saved LSM
        # that didn't have h_v_acc and h_s_acc

        # Verify that the loaded MNW is set up for the right task
        # 
        ut.abort_if(mnw.num_ctx != num_ctx, 'mnw.num_ctx != num_ctx')
        ctx_mod_mech = lsm_net.CtxModMech.from_str(args.ctx_mod_mech)
        ut.abort_if(mnw.ctx_mod_mech != ctx_mod_mech, 'mnw.ctx_mod_mech != ctx_mod_mech')
    else:
        mnw = lsm_net.Network('MNW', num_inputs, num_outputs, rng, args, num_ctx=num_ctx)

    # Save the untrained main network to file if requested
    #
    if (args.mnw_save_file is not None) and (mnw_file_mode == LsmFileMode.UNTRAINED):
        save_lsm(mnw, args.mnw_save_file)

    mnw.dump_topology()
    #mnw.dump_connectivity()

    if (ctx_mode in [CtxMode.RANDOM, CtxMode.EXPLICIT]) and not args.ctx_oracle:
        # Build or load the context identifier network.
        #
        if args.cid_load_file is not None:
            cid = load_lsm(args.cid_load_file, 'CID', num_inputs, num_ctx)
        else:
            cid = lsm_net.Network('CID', num_inputs, num_ctx, rng, args)

        # Save the untrained CID to file if requested
        #
        if (args.cid_save_file is not None) and (cid_file_mode == LsmFileMode.UNTRAINED):
            save_lsm(cid, args.cid_save_file)

        cid.dump_topology()
    else:
        cid = None

    # Train the network
    #
    train(mnw, cid, train_data, args.num_epochs)
    sep = mnw.get_sep()
    ut.info(f'separation = {sep:.03f}')
    
    # Save the trained MNW to file if requested
    #
    if (args.mnw_save_file is not None) and (mnw_file_mode == LsmFileMode.TRAINED):
        save_lsm(mnw, args.mnw_save_file)

    # Save the trained CID to file if requested
    #
    if (args.cid_save_file is not None) and (cid_file_mode == LsmFileMode.TRAINED):
        save_lsm(cid, args.cid_save_file)

    if args.separation_only:
        # Print sep as a single float number, as required by ga.py
        #
        print(f'{sep:.03f}')
    else:
        # Test it
        #
        acc = test(mnw, cid, test_data)
        ut.info(f'accuracy = {acc:.03f}')

        if args.print_separation:
            print(f'sep: {sep:.03f} acc: {acc:.03f}')
        else:
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
