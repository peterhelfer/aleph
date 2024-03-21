# Utility functions
#

#-------------------------------------------------------------------------------
# Copyright (C) 2019-2022  Peter Helfer
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

# Imports
#
import os, sys, enum, subprocess, argparse, time, csv, json, math, copy, datetime
from inspect import currentframe, getframeinfo
from scipy.special import expit
from scipy.special import logit
from scipy.spatial.distance import pdist
import scipy.stats
import matplotlib.pyplot as plt
import itertools as it
import numpy as np
import pandas as pd
import aenum

INFINITY = float('inf') # useful constant

def is_iterable(obj):
    """ Test if an object is iterable"""
    return hasattr(obj, '__iter__')

def is_listof(x, T):
    """test if x is a list of objects of type T (true if x is empty)"""
    return isinstance(x, list) and all(isinstance(element, T) for element in x)

def fmt_seq(fmt, seq, sep = ' '):
    "Format a sequence, applying fmt to each element"
    format_list = [fmt for item in seq]
    s = sep.join(format_list)
    return s.format(*seq)

def deep_fmt(fmt, seq, sep = ', ', nl_lvl=0, lvl=0):
    """Deep-format a sequence, applying fmt to each element, using sep as
    separator.
    :param fmt: format string
    :seq: the sequence to format
    :sep: string to use for separating elements
    :nl_lvl: terminate sub-sequences at this level and lower with newlines
    :lvl: used to track recursion level
    """
    if lvl <= nl_lvl:
        nl = '\n' + ' ' * (lvl+1)
    else:
        nl = ''
    
    if isinstance(seq, dict):
        lpar, rpar = '{', '}'
        isseq = True
    elif isinstance(seq, list):
        lpar, rpar = '[', ']'
        isseq = True
    elif isinstance(seq, tuple):
        lpar, rpar = '(', ')'
        isseq = True
    else:
        isseq = False
    
    if isseq:
        s = lpar
        for i, e in enumerate(seq):
            if i > 0:
                s += sep + nl
            s += deep_fmt(fmt, e, sep, nl_lvl=nl_lvl, lvl=lvl+1)

        s += rpar
    elif isinstance(seq, str):
        s = "'" + fmt.format(seq) + "'"
    else:
        s = fmt.format(seq)

    return s

def left_align(seq, width=None, pad=' '):
    """Append trailing pad characters to a sequence of strings to make 
    them all the same length.
    :param seq: A sequence of strings
    :param width: Length of the padded strings. By default, the length 
    of the longest string.
    :param pad: Pad character
    """
    if width is None: width = max(map(len, seq))
    return [s.ljust(width, pad) for s in seq]

def right_align(seq, width=None, pad=' '):
    """Prepend leading pad characters to a sequence of strings to make 
    them all the same length.
    :param seq: A sequence of strings
    :param width: Length of the padded strings. By default, the length
    of the longest string.
    :param pad: Pad character
    """
    if width is None: width = max(map(len, seq))
    return [s.rjust(width, pad) for s in seq]

def dot_align(seq, pad=' '):
    """Format a sequence of numbers as floating-point, padded with spaces on
    the left and right so as to align them on the decimal points and make
    the resulting strings be of equal lengths.
    :param seq: A sequence of numbers
    :param pad: Pad character
    """
    strs = [str(n) for n in seq]
    int_part_len = [len(s.split('.', 1)[0]) for s in strs]
    strs = [pad * (max(int_part_len) - d) + s for s, d in zip(strs, int_part_len)]
    ms = max(map(len, strs))
    return [s.ljust(ms, pad) for s in strs]

def dict_to_str(dict):
    """One-entry-per-line string representation of a dictionary"""
    s = ''
    for i, (name, value) in enumerate(dict.items()):
        if i > 0:
            s += '\n'
        s += f'{name}: {value}'
    return s

def print_dict(dict):
    """Print string representation of a dictionary"""
    print(dict_to_str(dict))

def dict_to_json(dict):
    """Return a json representation of a dictionary"""
    return json.dumps(dict, default=lambda o: o.__dict__, sort_keys=True, indent=2)

def dict_to_json_file(dict, fname):
    """Write a dictionary to file in json format"""
    with open(fname, 'w') as file:
        file.write(dict_to_json(dict))

def json_to_dict(s):
    """Convert json string to dict"""
    return json.loads(s)

def json_file_to_dict(fname):
    """Read dict from json format file"""
    with open(fname, 'r') as file:
        return json.loads(file.read())

def dict_to_csv_file(dict, fname):
    """Write a dictionary to file in CSV format"""
    with open(fname, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value])

def csv_file_to_dict(fname):
    """Read dict from CSV format file"""
    with open(fname) as csv_file:
        reader = csv.reader(csv_file)
        return dict(reader)

def find_indices(seq, cond=lambda x: bool(x)):
    """Get indices of items in seq for which cond is True"""
    return [i for i,x in enumerate(list) if cond(x)]

class StrEnum(aenum.Enum):
    """Enum where each constant has an associated string attribute,
    and conversion from string to enum constant is supported, 
    e.g. 
       class Fruit(StrEnum):
           APPLE = 'apple'
           APRICOT = 'apricot'

       my_fruit = Fruit.from_str('app')."""
    
    _init_ = 'value str'

    @classmethod
    def from_str(cls, str):
        """Convert from string to Enum constant. Case-insensitive.
        Accepts unique prefix of str attribute.
        """
        match = None
        for sc in list(cls):
            if sc.str.lower().startswith(str.lower()):
                if match is None:
                    match = sc
                else:
                    fatal(f'{get_loc(2)}: Non-unique StrEnum value: {str}')

        if match == None:
            fatal(f'{get_loc(2)}: Unknown StrEnum value: {str}')
            
        return match

# Trace: a simple tracing utility
#
class Trace(enum.IntEnum):
    """Trace levels"""
    FLOW = enum.auto()
    DEBUG = enum.auto()
    INFO3 = enum.auto()
    INFO2 = enum.auto()
    INFO = enum.auto()
    WARN = enum.auto()
    ERROR = enum.auto()
    FATAL = enum.auto()

trace_level = Trace.INFO

def set_trace_level(level):
    global trace_level
    trace_level = level

def set_trace_level_str(level):
    level = level.upper()
    if   level == 'FLOW':  set_trace_level(Trace.FLOW)
    elif level == 'DEBUG': set_trace_level(Trace.DEBUG)
    elif level == 'INFO3': set_trace_level(Trace.INFO3)
    elif level == 'INFO2': set_trace_level(Trace.INFO2)
    elif level == 'INFO':  set_trace_level(Trace.INFO)
    elif level == 'WARN':  set_trace_level(Trace.WARN)
    elif level == 'ERROR': set_trace_level(Trace.ERROR)
    elif level == 'FATAL': set_trace_level(Trace.FATAL)
    else:
        print(f"Unknown trace level '{level}'")
        sys.exit(2)

trace_file = sys.stdout

def set_trace_file(file):
    global trace_file
    trace_file = file

def trace(level, str, notime=False):
    """Print a trace message.
       :param level: Print str if trace_level >= level
       :param str: Message to print
       :param notime: Supress timestamp
    """
    if (level >= trace_level):
        if level >= Trace.WARN:
            if notime:
                eprint(str)
            else:
                teprint(str)
        else:
            if notime:
                print(str, file=trace_file)
            else:
                tprint(str, file=trace_file)
            trace_file.flush()
    if level == Trace.FATAL:
        sys.exit(2)

def flow(str, **kwargs):  trace(Trace.FLOW,  'FLOW: ' +str, **kwargs)
def debug(str, **kwargs): trace(Trace.DEBUG, 'DEBUG: ' +str, **kwargs)
def info3(str, **kwargs): trace(Trace.INFO3, 'INFO3: ' +str, **kwargs)
def info2(str, **kwargs): trace(Trace.INFO2, 'INFO2: ' +str, **kwargs)
def info(str, **kwargs):  trace(Trace.INFO,  'INFO: ' +str, **kwargs)
def warn(str, **kwargs):  trace(Trace.WARN,  'WARN: ' +str, **kwargs)
def error(str, **kwargs): trace(Trace.ERROR, 'ERROR: ' +str, **kwargs)
def fatal(str, **kwargs): trace(Trace.FATAL, get_loc(3) + ' FATAL: ' + str, **kwargs)
def abort_if(cond, str, **kwargs):
    if cond: fatal(str, **kwargs)
def abort_unless(cond, str, **kwargs):
    abort_if(not cond, str, **kwargs)

# String functions
#
def strcmp(s1, s2, n=float('inf'), ci=False, prefix=False):
    """String compare with options
    :param s1: String 1
    :param s2: String 2
    :param n: Max num chars to compare
    :param ci: case-independent
    :param prefix: compare up to shorter of s1, s2 length
    :return: -1/0/+1 for s1 </==/> s2"""
    if prefix:
        n = min(n, len(s1), len(s2))
        
    s1 = s1[:n] if n < len(s1) else s1
    s2 = s2[:n] if n < len(s2) else s2
    if ci:
        s1 = s1.lower()
        s2 = s2.lower()
    return -1 if s1 < s2 else 1 if s1 > s2 else 0

def ci_strcmp(s1, s2, **kwargs):
    """Case-independent string compare"""
    return strcmp(s1, s2, ci=True, **kwargs)

def streq(s1, s2, **kwargs):
    """String equality test"""
    return strcmp(s1, s2, **kwargs) == 0

def ci_streq(s1, s2, **kwargs):
    """Case-independent string equality test"""
    return streq(s1, s2, ci=True,**kwargs)

def is_prefix_of(s1, s2, **kwargs):
    """Test if s1 is a prefix of s2"""
    return streq(s1, s2, n=len(s1), **kwargs)


def csv_to_list(csv, candidates, prefix=True, ci=True):
    """Construct a list of strings based on a csv string.
    Include in the list all candidates that are matched by elements of csv.
    If csv == "all", include all the candidates
    :param csv: Comma-separated list of strings
    :param candidates: Set of candidate values to include in list
    :param prefix: Whether to accept prefix matches
    :param ci: Use case-independent matching"""

    if ci_streq(csv, 'all'):
        return candidates

    lst = []
    for s in csv.split(','):
        s_found = False
        for c in candidates:
            if streq(s, c, prefix=prefix, ci=ci):
                lst += [c]
                s_found = True
        if not s_found:
            fatal(f'"{s}" not in "{candidates}"')
    return lst

# Timestamp functions
#
def datetime_str(dt, dsep='_', dtsep='__', tsep='_'):
    """Build a datetime string like yyyy_mm_dd__hh_mm_ss
    :param dt: a datetime.datetime object
    :param dsep: separator between year, month and day
    :param dtsep: separator between  date and time parts
    :param tsep: separator between jour, minutes and seconds"""
    return (f'{dt.year}{dsep}{dt.month:02d}{dsep}{dt.day:02d}{dtsep}'
            f'{dt.hour:02d}{tsep}{dt.minute:02d}{tsep}{dt.second:02d}')

#def now_str(dsep='_', dtsep='__', tsep='_'):
def now_str(**kwargs):
    """Build a string representation of current date and time"""
    return datetime_str(datetime.datetime.now(), **kwargs)

def hms(seconds):
    """Convert seconds to hours, minutes, seconds"""
    h = int(seconds / 3600)
    m = int(seconds / 60) - 60 * h
    s = seconds - 3600 * h - 60 * m
    return h, m, s

def hms_str(seconds):
    """Convert seconds to hh:mm:ss string"""
    h, m, s = hms(seconds)
    return(f'{h:02d}:{m:02d}:{s:05.2f}')

start_time = time.time()

def elapsed_str():
    """Return elapsed execution time in hh:mm:ss format"""
    return hms_str(time.time() - start_time)

def tprint(str, file=sys.stdout):
    """Print str on stdout preceded by timestamp"""
    print(f'{elapsed_str()}  {str}', file=file)

def teprint(str):    
    """Print str on stderr preceded by timestamp"""
    tprint(str, file=sys.stderr)

def eprint(*args, **kwargs):
    """Print on stderr"""
    print(*args, file=sys.stderr, **kwargs)

def make_random_dict(m, n):
    """Create a dictionary that randomly maps integer keys
    in the range [0:m[ to values in[0:n[, mapping an equal number
    of keys (as near as possible) to each value.
    TODO: generalize to arbitrary data types."""
    dict = {}
    keys = np.random.permutation(m)
    val = 0
    for key in keys:
        dict[key] = val
        val = (val + 1) % n
    return dict

def run_cmd(cmd, timeout=None, capture_output=True):
    """Run a shell command and wait for it to finish
    :param cmd: The command to run
    :return: stdout from the command"""
    p = subprocess.run(cmd, shell=True, text=True,
                       capture_output=capture_output,
                       timeout=timeout)
    return p.stdout

# Get command that invoked current process
#   Can't undo shell's munging (e.g. variable interpolation),
#   but otherwise pretty close. Should reproduce argv if typed
#   on the command line.
#
def get_cmd(pname='full'):
    cmdline = ''
    for i, arg in enumerate(sys.argv[:]):
        if cmdline != '':
            cmdline += ' '
        if len(arg.split()) > 1:
            arg = "'" + arg + "'"
        if i == 0:
            if pname == 'none':
                continue
            elif pname == 'base':
                arg = os.path.basename(arg)
        cmdline += arg
    return cmdline

# Raise error if an argument is not a positive int
#
def check_pos(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is not a valid positive int value" % value)
    return ivalue

# Raise error if an argument is not a non-negative int
#
def check_nonneg(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("%s is not a valid non-negative int value" % value)
    return ivalue

# ArgumentParser with terser (positional) add_argument syntax
#
class TerseArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(TerseArgumentParser, self).__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)

    def add_int_arg(self, name, default, help, **kwargs):
        self.add_argument('--'+name, type=int, default=default, metavar='INT', help=help, **kwargs)

    def add_int_arg2(self, short_name, name, default, help, **kwargs):
        self.add_argument('-'+short_name, '--'+name, type=int, default=default, metavar='INT', help=help, **kwargs)

    def add_nonneg_int_arg(self, name, default, help, **kwargs):
        self.add_argument('--'+name, type=check_nonneg, default=default, metavar='INT', help=help, **kwargs)

    def add_nonneg_int_arg2(self, short_name, name, default, help, **kwargs):
        self.add_argument('-'+short_name, '--'+name, type=check_nonneg, default=default, metavar='INT', help=help, **kwargs)

    def add_pos_int_arg(self, name, default, help, **kwargs):
        self.add_argument('--'+name, type=check_pos, default=default, metavar='INT', help=help, **kwargs)

    def add_pos_int_arg2(self, short_name, name, default, help, **kwargs):
        self.add_argument('-'+short_name, '--'+name, type=check_pos, default=default, metavar='INT', help=help, **kwargs)

    def add_float_arg(self, name, default, help, **kwargs):
        self.add_argument('--'+name, type=float, default=default, metavar='FLOAT', help=help, **kwargs)
        
    def add_float_arg2(self, short_name, name, default, help, **kwargs):
        self.add_argument('-'+short_name, '--'+name, type=float, default=default, metavar='FLOAT', help=help, **kwargs)
        
    def add_str_arg(self, name, default, help, required=False, **kwargs):
        self.add_argument('--'+name, type=str, default=default, metavar='STRING', help=help, required=required, **kwargs)
        
    def add_str_arg2(self, short_name, name, default, help, required=False, **kwargs):
                self.add_argument('-'+short_name, '--'+name, type=str, default=default, metavar='STRING', help=help, required=required, **kwargs)

    def add_bool_arg(self, name, help, **kwargs):
        self.add_argument('--'+name, action='store_true', help=help, **kwargs)
        
    def add_bool_arg2(self, short_name, long_name, help, **kwargs):
        self.add_argument('-'+short_name, '--'+long_name, action='store_true', help=help, **kwargs)

    def add_onoff_arg(self, on_name, off_name, default, help, **kwargs):
        self.add_argument('--'+on_name, default=default, action='store_true', help=help, **kwargs)
        self.add_argument('--'+off_name, dest=on_name, default=default, action='store_false', help=f'Set {on_name} to False', **kwargs)
        #self.set_defaults(on_name=default)

class RunningStats:
    "Compute running mean and stdev for a series of float values"
    def __init__(self):
        self.n = 0
        self.m = 0
        self.new_m = 0
        self.s = 0
        self.new_s = 0
        self.mean_ = 0
        self.variance_ = 0
        self.stdev_ = 0
        self.is_dirty = True
        
    def clear(self):
        self.n = 0
    
    def push(self, x):
        self.n += 1
    
        if self.n == 1:
            self.m = self.new_m = x
            self.s = 0
        else:
            self.new_m = self.m + (x - self.m) / self.n
            self.new_s = self.s + (x - self.m) * (x - self.new_m)
        
            self.m = self.new_m
            self.s = self.new_s
        self.is_dirty = True

    def update(self):
        self.mean_ = self.new_m if self.n else 0.0
        self.variance_ = self.new_s / (self.n - 1) if self.n > 1 else 0.0
        self.stdev_ = math.sqrt(self.variance_)
        self.is_dirty = False
        
    def mean(self):
        if self.is_dirty: self.update()
        return self.mean_
    
    def variance(self):
        if self.is_dirty: self.update()
        return self.variance_
    
    def stdev(self):
        if self.is_dirty: self.update()
        return self.stdev_

# Logistic (asymmetric sigmoid) function
#
def logistic(x, k=1, x_half=0):
    """Asynchronous sigmoid function
         Parameters:
            x: The variable
            k: Higher value gives steeper slope
            x_half: The x-value where the asigmoid crosses y=0.5
       To concentrate almost all the action in the [0,1] interval,
       try k=10, x_half=0.5."""

    return 1 / (1 + np.exp(-k * (x - x_half)));

def asigmoid(x, k=1, x_half=0):
    return logistic(x, k=1, x_half=0)

# Sigmoid function
#
def sigmoid(x, k=1, x_zero=0):
    """Sigmoid function
         Parameters:
            x: The variable
            k: Higher value gives steeper slope
            x_zero: The x-value where the sigmoid crosses y=0.0
       To concentrate almost all the action in the [0,1] interval,
       try k=10, x_zero=0.5."""
    return asigmoid(x, k, x_zero) - 0.5

def std_score(x):
    """Calculate standard score (a.k.a. Z-score) of numpy vector x"""
    assert x.ndim == 1
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0:
        return x-mu
    else:
        return (x - mu) / sigma

def scale(x, old_xmin, old_xmax, new_xmin=0, new_xmax=1):
    """Scale x from [old_xmin, old_xmax] to [new_xmin, new_xmax]"""
    fraction = (x - old_xmin) / (old_xmax - old_xmin)
    return new_xmin + fraction * (new_xmax - new_xmin)

def scaled_logit(x, xmin=None, xmax=None, ymin=0, ymax=1, marg=0.00001):
    """Apply the logit function to x after mapping it from [xmin, xmax]
    to [marg, 1-marg], and scale the result from logit([marg, 1-marg])
    to [ymin, ymax]. A non-zero marg avoids infinity at xmin, xmax.
    """
    if xmin == None: xmin = x.min()
    if xmax == None: xmax = x.max()

    x = scale(x, xmin, xmax)
    x = marg + x * ((1 - marg) - marg) # map x to [marg, 1-marg]

    v = logit(x)

    # scale y to [ymin, ymax]

    vmin = logit(marg)
    vmax = logit(1-marg)
    y = ymin + (v - vmin) / (vmax - vmin) * (ymax - ymin)

    return y

def sharpen(x, k=1, mid=0.5):
    """Apply a logistic sigmoid to the elements of a numpy array,
       such that the min and max values in x are mapped to themselves
       and the mid-point of the sigmoid falls at the mid fraction of
       the x range. 
       Parameters:
         x: a numpy array
         k = steepness of the logistic
         mid: relative position of the sigmoid mid-point"""

    x_range = x.max() - x.min()
    return x.min() + x_range * expit(k * ((x - x.min()) / x_range - mid))

def test_sharpen():
    x_min = 0
    x_max = 200

    x_values = np.arange(x_min, x_max, (x_max - x_min) / 50)
    y_values = sharpen(x_values, 10, 0.7)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(x_values, y_values, 'r')
    plt.show()

def pattern_separation(patterns, labels):
    """Compute the separation property of the patterns as the ratio of mean
    inter-class distance to mean intra-class distance.  
    :param patterns: sequence of equal-length vectors
    :param labels: sequence of class labels for the vectors
    :return: separation, mean_inter_class_distance, mean_intra_class_variance:
    """

    assert len(patterns) == len(labels)
    
    # Inter-class distance is the mean of the pair-wise euclidean distances
    # between classes' centroids (averaged patterns).

    # Intra-class variance is the mean of the euclidean distances of a class's
    # patterns from the class's average pattern.

    # Convert to ndarrays in case they aren't already
    #
    patterns = np.array(patterns)
    labels = np.array(labels)
    
    centroids = {}
    mean_dist = {}

    for lbl in np.unique(labels):
        pats = patterns[np.where(labels == lbl)]

        # Mean of the pats 
        centroids[lbl] = np.mean(pats, axis=0)

        # Mean dist between pats
        if len(pats) > 1:
            # Norton&Ventura divide the mean distance between centroids by
            # the mean intra-class variance:
            # mean_dist[lbl] = np.mean(np.linalg.norm(pats - centroids[lbl], axis=1))
            # but I think it makes more sense to divide mean distance by mean distance
            # (or variance by variance) as a measure of overlap:
            mean_dist[lbl] = np.mean(pdist(pats))
        else:
            mean_dist[lbl] = 0

    mean_intra_class_distance = np.mean(list(mean_dist.values()))
    mean_inter_class_distance = np.mean(pdist(list(centroids.values())))
    separation = mean_inter_class_distance / mean_intra_class_distance
    return(separation, mean_inter_class_distance, mean_intra_class_distance)

def test_pattern_separation():
    patterns = [[4, 5, 6], [110, 8, 9], [110,11,12]]
    labels = ['a','b','b']
    sep, inter, intra = pattern_separation(patterns, labels)
    print(f'sep={sep:.02f}, inter={inter:.02f}, intra={intra:.02f}')

#test_pattern_separation()

def unique_unordered_pairs(seq):
    """
    :param seq: A sequence of comparable items
    :return: A sequence of all unique unordered pairs of elements in seq:
    """
    return [p for p in it.product(seq, seq) if p[0] < p[1]]

class SpikeNorm():
    """Dummy normal distribution with zero variance"""
    def __init__(self, mu):
        self.mu = mu
    def rvs(self):
        return self.mu

def truncnorm(mu, sigma, lower, upper):
    """A truncated normal continuous random variable.
    This is a wrapper around scipy.stats.truncnorm, which requires
    clip values in the domain of the standard normal. This truncnorm
    accepts lower/upper parameters as the actual clip value.
    Also, this truncnorm accepts sigma==0, by returning an object
    whose rvs method simply returns mu.
    :param mu: Mean value
    :param sigma: Standard deviation
    :param lower: Lower clip value
    :param upper: Upper clip value
    """
    if sigma == 0:
        return SpikeNorm(mu)
    else:
        a = (lower - mu) / sigma
        b = (upper - mu) / sigma
        return scipy.stats.truncnorm(a, b, loc=mu, scale=sigma)

def fast_truncnorm(rng, mu, sigma, lower, upper):
    """Same as truncnorm above, but (a) using a simple rejection approach,
    (b) drawing and returning a value instead of returning a variable.
    Note: the slowness of truncnorm is a known problem, 
    see https://github.com/scipy/scipy/issues/12733
    As of scipy 1.9.0, fast_truncnorm is >100 times faster than truncnorm.
    """
    if sigma == 0:
        return mu
    else:
        while True:
            v = rng.normal(loc=mu, scale=sigma)
            if lower <= v < upper:
                return v

v_fast_truncnorm = np.vectorize(fast_truncnorm)
    
class FastUniform():
    """Buffering wrapper for rng.uniform(). Get bufsize random numbers at a time
    from rng.uniform, then dole them out one or a few at a time in the get()
    method.  This is much faster than making many calls to rng.uniform(), which
    is nice when vectorizing is not possible. e.g. in a loop.
    """
    def __init__(self, rng, bufsize):
        self.rng = rng
        self.bufsize = bufsize
        self.i = bufsize

    def get(self, size=None):
        """Get one or more uniform random numbers.
        :param size: If None, return a scalar, otherwise a vector of size size
        """
        n = 1 if size is None else size
        if self.i + n > self.bufsize:
            self.buf = self.rng.uniform(size=self.bufsize)
            self.i = 0
            
        if size is None:
            r = self.buf[self.i]
        else:
            r = self.buf[self.i:self.i+size]
        self.i += n
        return r

def heatmap(data, annotate, title='', xlabel='', ylabel='',
            row_labels=None, col_labels=None, percent=None,
            lighttextcolor="white", darktextcolor="black",
            rows_up=False, **kwargs):
    """Plot a 2D Numpy array as a heatmap
    :param data: The array
    :param annotate: Whether to print cell values 
    :param title: Plot title
    :param xlabel: X-axis label
    :param ylabel: Y-axis label
    :param percent: If 'row' or 'col', then display cell values as percentages of 
                    row/column sums
    :param rows_up: Display rows from bottom to top
    """
    def safediv(x, y):
        """Divide x by y, but if x is zero, return zero.
        This is so that all-zero columns (or rows) be shown as
        0, not as naN in percent representation."""
        if x == 0:
            return 0
        else:
            return x / y
    vsafediv = np.vectorize(safediv)

    fig, ax = plt.subplots()

    if rows_up:
        data = np.flip(data, axis=0)
        row_labels = np.flip(row_labels)
    else:
        ax.get_xaxis().tick_top()
        ax.get_xaxis().set_label_position('top') 

    if percent=='col':
        data = np.round(vsafediv(100 * data, np.sum(data, axis=0))).astype(int)
    elif percent=='row':
        data = np.round(vsafediv(100 * data / np.sum(data, axis=1))[:,None]).astype(int)

    #im = ax.imshow(data.round(decimals=2), cmap='bone')
    im = ax.imshow(data, cmap='bone')

    if annotate:
        medval = (np.min(data) + np.max(data)) / 2
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val < medval:
                    textcolor = lighttextcolor
                else:
                    textcolor = darktextcolor
                ax.text(col, row, data[row, col],
                        ha="center", va="center", color=textcolor)

    ax.set_title(title,loc='left',ha='left')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    if row_labels is not None:
        ax.set_yticklabels(row_labels)

    if col_labels is not None:
        ax.set_xticklabels(col_labels)

    plt.show(**kwargs)

# Confusion matrix
#
class ConfusionMatrix():
    def __init__(self, size=None, row_labels=None, col_labels=None, title='',
                 percent=None,
                 xlabel='x', ylabel='y',
                 lighttextcolor='white', darktextcolor='black', value_range=None):
        """
        If value_range is given, than the confusion matrix will show rows and columns
        for the values in that range. Otherwise, rows and columns will only be shown for
        values that occur in the data. Example: if the only data points are (2,3) and (6,8),
        then the matrix will have four rows and four columns.
        :param size: Unused legacy param
        :param row_labels: Row labels
        :param col_labels: Column labels
        :param title: Chart title
        :param percent: If =='row' or =='col', then display cell values as 
                        percentages of row/column sums
        :param xlabel: X-axis label
        :param ylabel: Y-axis label
        :param lighttextcolor: Used for text on dark squares
        :param darktextcolor: Used for text on light squares
        :param value_range: row/column values (may differ from labels)"""

        self.row_labels = row_labels
        self.col_labels = col_labels
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.percent = percent
        self.title = title
        self.lighttextcolor = lighttextcolor
        self.darktextcolor = darktextcolor
        self.value_range = value_range
        self.values = np.zeros((0, 2), dtype=int)

    def add_value(self, row, col):
        """Add an x/y pair"""
        self.values = np.append(self.values, np.array([[row, col]]), axis=0)

    def set_values(self, row_values, col_values):
        """Add a bunch of row/col pairs"""
        for (row, col) in zip(row_values, col_values):
            self.add_value(row, col)

    def incr(self, row, col):   # Legacy synonym for add_value
        self.add_value(row, col)
                       
    def plot(self, **kwargs):
        """Plot the matrix."""

        # Build a map from values to row/column indices
        #
        if self.value_range is not None:
            # The matrix has all the values in the range
            indices = {v:v for v in self.value_range}
        else:
            # The matrix has rows/columns only for values we have seen
            uvals = np.unique(self.values)
            indices = {value:index for index, value in enumerate(uvals)}

        data = np.zeros((len(indices), len(indices)), dtype=int)
        for pair in self.values:
            data[indices[pair[0]], indices[pair[1]]] += 1
        if self.row_labels  is None: self.row_labels = list(indices.keys())
        if self.col_labels  is None: self.col_labels = list(indices.keys())

        abort_if(len(self.row_labels) != len(indices), 'Wrong number of row labels')
        abort_if(len(self.col_labels) != len(indices), 'Wrong number of col labels')

        heatmap(data,
                True, self.title, percent=self.percent,
                row_labels=self.row_labels,
                col_labels=self.col_labels,
                xlabel=self.xlabel, ylabel=self.ylabel,
                lighttextcolor=self.lighttextcolor,
                darktextcolor=self.darktextcolor, **kwargs)

def shift(arr, n, fill_value=np.nan):
    """Shift a numpy array by n.
    :param arr: Array to shift.
    :param n: How far to shift (positive to the right).
    :param fill_value: Value "shifted in" from the left or right."""
    result = np.empty_like(arr)
    if n > 0:
        result[:n] = fill_value
        result[n:] = arr[:-n]
    elif n < 0:
        result[n:] = fill_value
        result[:n] = arr[-n:]
    else:
        result[:] = arr
    return result

def cosine_sim(v1, v2):
    """Calculate the cosine similarity between two numpy vectors.
    :param v1: the first vector
    :param v2: the second vector
    :return: The cosine similarity between v1 and v2"""
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d == 0:
        return 0 # avoid div-by-zero
    else:
        return np.dot(v1, v2)/d
    
def moving_average1d(arr, w):
    """Calculate the moving average of a 1D numpy array so that 
    ret[i] is the average of the preceding i values of arr,
    in other words,
    ret[i] = 0,                   if i < w
    ret[i] = sum(arr[i-w:i-1])/w, otherwise
    :param arr: The array.
    :param w: window over which to calculate the average."""
    abort_if(w > len(arr), 'Moving window longer than array.')
    conv = np.convolve(arr, np.ones(w), mode='valid') / w
    return np.pad(conv[:len(arr)], (w, 0))[:-1]

def moving_average2d(arr, w, axis=0):
    """Calculate the moving average for each row (if axis=0) 
    or column of arr.
    :param arr: The array.
    :param w: window over which to calculate the average."""
    ret = np.zeros_like(arr)
    if axis == 0:
        for col in range(arr.shape[1]):
            ret[:, col] = moving_average1d(arr[:, col].T, w).T
    elif axis == 1:
        for row in range(arr.shape[0]):
            ret[row,:] = moving_average1d(arr[row,:], w)
    else:
        fatal(f'invalid axis: {axis}')
    return ret

def plot(x, y, xlabel='X', ylabel='Y'):
    """Plot a simple x/y graph."""

    #plt.figure(1)
    #plt.clf()

    fig, ax = plt.subplots()

    ax.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show(block=True)


def plot_lines(x, y, log=False, scale=None, title=None,
               xlabel='points', ylabel='values',
               series_labels=None,
               colors=None, linestyles=None,
               vline_x=None, block=False, **kwargs):
    """Plot some lines.
    :param x: sequence of x-values
    :param y: numpy array of y-values, one column per series
    :param log: Use logarithmic y-axis
    :param scale: Map each series to given scale [min, max]
    :param title: Plot title.
    :param xlabel: X-axis label.
    :param ylabel: Y-axis label.
    :param series_labels: Series labels (for legend).
    :param vline_x: Draw a vertical line at this x value.
    """
    y = copy.deepcopy(y) # Make local copy that we can mess with

    if series_labels is None:
        series_labels = [None] * y.shape[1]

    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111)
    if log:
        ax.set_yscale('log')
    
    if scale is not None:
        assert np.array(scale).shape == (2,)
        for band in range(1, y.shape[1]):
            min = np.min(y[:,band])
            max = np.max(y[:,band])
            if min != max:
                y[:,band] = np.interp(y[:,band],
                                      (min, max),
                                      scale)
    
    for c in range(y.shape[1]):
        line, = ax.plot(x, y[:, c], label=series_labels[c], **kwargs)

    if linestyles is not None:
        for i, linestyle in enumerate(linestyles):
            ax.lines[i].set_linestyle(linestyle)

    if colors is not None:
        for i, color in enumerate(colors):
            ax.lines[i].set_color(color)

    if vline_x:
        plt.axvline(x=vline_x, c='k', ls='--')
    
    plt.title(title)
    ax.set_xlabel(xlabel)#, loc='right')
    ax.set_ylabel(ylabel)#, loc='top')
    
    # Shrink current axis's height by 15% on the bottom to make room for legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.25,
                     box.width * 0.9, box.height * 0.75])
    
    # Put a legend below the x-axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07),
              ncol=6, fontsize='x-small')
    
    plt.show(block=block)
    #plt.close()

def pad_array(a, shape, loc=(0,0), value=0):
    """Expand a 2D numpy array by padding with a specified value.
    :param a: The original array.
    :param shape: The shape of the padded array.
    :param loc: location in the padded array of a[0,0]
    :return: the padded array"""
    b = np.full(shape, value)
    b[loc[0]:loc[0] + a.shape[0],loc[1]:loc[1] + a.shape[1]] = a
    return b

def pd_print_full(x):
    """Print a Pandas table without ellipses (...)"""
    values 
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def np_print_full(*args, **kwargs):
  """Print a Numpy array without ellipses (...)"""
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold=numpy.inf)
  print(*args, **kwargs)
  numpy.set_printoptions(**opt)    

def get_loc(level=1):
    """Return file name and line number of caller.
    :param level: use caller this many stack frames up"""
    
    frame = currentframe()
    for i in range(level):
        if frame.f_back is not None:
            frame = frame.f_back
    
    frameinfo = getframeinfo(frame)
    return f'{os.path.basename(frameinfo.filename)}:{frameinfo.lineno}'

def make_file_name(part1=None, sep='_', part2=None, ext='.tmp'):
    """Make a file name like part1_part2.ext, with default
    pname_yyyy_mm_dd__hh_mm_ss.tmp
    """
    if part1 is None:
        part1 = get_cmd('base').split(' ')[0].split('.')[0]
    if part2 is None:
        part2 = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

    return part1 + sep + part2 + ext
