# Types used in test datasets for LSM and functions for creating such datasets
#

import math, random, enum, gzip, pickle
import numpy as np

import util as ut

# Some definitions
#
# Patterns 
#
# pattern     = a sequence of firing rates
# pattern_set = a list of patterns, one per channel
# labeled_pattern_set = a tuple of a context_id (int), a label (int) and a pattern_set
# label       = an int that identifies a class of patterns
# context     = an int; when multiple contexts are used, pattern_set-label associations
#               are context-specific.

# labeled_pattern_sets are used to generate samples:

# block   = list of ints representing time frames with (1) or without (0) spikes,
#           generated with some specified average firing rate
# stream  = concatenation of blocks with specified spiking rates,
#           all with the same specified block length and time frame length
# sample  = a list of streams, one per channel
# labeled_sample = a tuple of a context, a label and a sample
# dataset = a list of labeled_samples

class LabeledPatternSet:
    def __init__(self, context, label, pattern_set):
        self.context = context
        self.label = label
        self.pattern_set = pattern_set # list of lists of mean spiking frequencies

class LabeledSample:
    def __init__(self, context, label, sample):
        self.context = context
        self.label = label
        self.sample = sample # list of lists of zeros and ones representing spike/no-spike time frames

class Dataset:
    def __init__(self, description, block_size, labeled_samples, labeled_pattern_sets=None):
        self.description = description
        self.block_size = block_size
        self.labeled_samples = labeled_samples
        self.labeled_pattern_sets = labeled_pattern_sets

    def get_pattern_separation_v2(self):
        """ Experimental. This only works when the patterns have length 1.
        """
        patterns = [lps.pattern_set for lps in self.labeled_pattern_sets]
        labels = [lps.label for lps in self.labeled_pattern_sets]
        return ut.pattern_separation(np.array(patterns)[:,:,0], labels)

    # I'm not sure this function makes any sense. It calculates distances between
    # 2-dimensional arrays. 

    def get_pattern_separation(self):
        """Compute the separation property of the pattern_sets as the ratio of
        mean_inter_class_distance to mean_intra_class_variance.

        Note: the mean_intra_class-variance will (of course) only be
        non-zero when there are multiple pattern_sets per class (label).
        :return: separation
        :return: mean_inter_class_distance
        :return: mean_intra_class_variance
        """
        # mean_inter_class_distance is the mean of the norms of
        # the pair-wise differences between classes' centers of mass.
        #
        # mean_intra_class_variance is the mean of norms of the
        # differences of a class's patterns from the class's center
        # of mass.

        # Calculate the centers of mass for each class's patterns.
        #
        classes = list(set([ls.label for ls in self.labeled_pattern_sets]))
        num_classes = len(classes)
        pattern_sets = [None] * num_classes
        pattern_set_centers_of_mass = [None] * num_classes

        for cl in range(len(classes)):
            pattern_sets[cl] = np.array([lps.pattern_set
                                         for lps in self.labeled_pattern_sets
                                         if lps.label == classes[cl]])
            pattern_set_centers_of_mass[cl] = np.mean(pattern_sets[cl], axis=0)

        # Then, calculate the mean distance between pairs of
        # pattern_set_centers_of_mass.
        #
        mean_inter_class_distance = (
            np.mean([np.linalg.norm(pattern_set_centers_of_mass[i] -
                                    pattern_set_centers_of_mass[j])
                     for i in range(num_classes)
                     for j in range(num_classes)
                     if i < j]))

        # mean_intra_class_variance:
        #   For each class, calculate the variance of the pattern_sets.
        #   Then take the average of those.
        #
        intra_class_variances = \
            [np.sum(np.linalg.norm(pattern_sets[i] -
                                   pattern_set_centers_of_mass[i], axis=1)) /
             len(pattern_sets[i])
             for i in range(num_classes)]
        mean_intra_class_variance = np.mean(intra_class_variances)

        if mean_intra_class_variance == 0:
            separation = float('inf')
        else:
            separation = mean_inter_class_distance / mean_intra_class_variance

        return separation, mean_inter_class_distance, mean_intra_class_variance


    def get_sample_separation(self):
        """Compute the separation property of the samples as the ratio of
        mean_inter_class_distance to mean_intra_class_variance.
        :return: separation
        :return: mean_inter_class_distance
        :return: mean_intra_class_variance
        """
        # mean_inter_class_distance is the mean of the norms of
        # the pair-wise differences between classes' centers of mass.
        #
        # mean_intra_class_variance is the mean of norms of the
        # differences of a class's samples from the class's center
        # of mass.

        # Calculate the centers of mass for each class's samples.
        #
        classes = list(set([ls.label for ls in self.labeled_samples]))
        num_classes = len(classes)
        samples = [None] * num_classes
        sample_centers_of_mass = [None] * num_classes

        for cl in range(num_classes):
            samples[cl] = np.array([ls.sample for ls in self.labeled_samples
                                    if ls.label == classes[cl]])
            sample_centers_of_mass[cl] = np.mean(samples[cl], axis=0)

        # Then, calculate the mean distance between pairs of sample_centers_of_mass.
        #
        mean_inter_class_distance = (
            np.mean([np.linalg.norm(sample_centers_of_mass[i] - sample_centers_of_mass[j])
                   for i in range(num_classes) for j in range(num_classes) if i < j]))

        # mean_intra_class_variance:
        #   For each class, calculate the variance of the samples.
        #   Then take the average of those.
        #
        intra_class_variances = [np.sum(np.linalg.norm(samples[i] - sample_centers_of_mass[i], axis=1)) /
                                 len(samples[i]) for i in range(num_classes)]
        mean_intra_class_variance = np.mean(intra_class_variances)

        if mean_intra_class_variance == 0:
            separation = float('inf')
        else:
            separation = mean_inter_class_distance / mean_intra_class_variance

        return separation, mean_inter_class_distance, mean_intra_class_variance

# Generator types
#
class Generator(ut.StrEnum):
    POISSON = 'poisson'
    FIXED_RATE = 'fixed_rate'
    DETERMINISTIC = 'deterministic'
    RATE_BASED = 'rate_based'

generator_type = Generator.POISSON # default

def set_generator_type(type):
    global generator_type
    generator_type = type

class ContextSpec:
    """A triplet that specifies which label a particular pattern should be
    associated with in a given context."""
    def __init__(self, ctx, pattern_id, label):
        """
        :param ctx: a context id
        :param pattern_id: a pattern id
        :param label: the label associated with this pattern in this context"""
        self.ctx = ctx
        self.pattern_id = pattern_id
        self.label = label



# Functions for building elements of a dataset
#
def make_block(block_size, frame_length, firing_rate):
    """Generate a list of spike/no-spike time frames (zeros and ones) for the
    given block_size and frame_length (msec). If generator_type is
    POISSON, then use a per-frame spike probability corresponding to the
    given firing_rate (Hz).  If it is FIXED_RATE, then simply emit spikes at
    the given rate, starting at a random point in the cycle. DETERMINISTIC
    is like FIXED_RATE, but the first spike is always in the first frame. If
    generator_type is RATE_BASED, then each frame is set to the value of
    the firing probability, optionally with some noise added.
    :param block_size: Length of a block in frames.
    :param frame_length: Length of a frame in milliseconds.
    :param firing_rate: Firing rate in Hz
    :return: List of ints
    """
    
    # Calculate per-frame firing probability
    #
    p = firing_rate * frame_length / 1000.0 # length is msec, rate is Hz

    # Average interval (number of frames) between spikes
    #
    if p == 0:
        intrvl = ut.INFINITY
    else:
        intrvl = 1 / p
        
    if generator_type == Generator.POISSON:
        return (np.random.random(int(block_size)) < p).astype(int).tolist()
    elif generator_type == Generator.FIXED_RATE:
        block = [0] * block_size
        t = np.random.randint(intrvl)
        while t < block_size:
            block[int(t)] = 1
            t += intrvl
        return block
    elif generator_type == Generator.DETERMINISTIC:
        block = [0] * block_size
        t = 0
        while t < block_size:
            block[int(t)] = 1
            t += intrvl
        return block
    elif generator_type == Generator.RATE_BASED:
        # Fill the block with values p ± 10%
        block = [random.uniform(0.9 * p, 1.1 * p) for i in range(block_size)]
        return block
    else:
        ut.fatal("Unknown generator_type")

def make_stream(block_size, frame_length, firing_rates):
    """Generate a stream as a concatenation of blocks with the given
    firing_rates"""
    stream = []
    for fr in firing_rates:
        stream += make_block(block_size, frame_length, fr)
    return stream

def make_random_pattern_set(num_channels, num_blocks, min_rate, max_rate):
    """Generate a list of patterns. Each pattern is a list of firing rates
    drawn uniformly from [min_rate, max_rate]."""
    ps = [np.random.randint(min_rate, max_rate + 1, num_blocks).tolist()
          for i in range(num_channels)]
    return ps

def make_sample(block_size, frame_length, pattern_set):
    """Generate a list of streams according to the given pattern_set. Each
    pattern is a list of firing rates."""
    # Verify that the patterns have equal length
    for p in pattern_set[1:]:
        assert len(p) == len(pattern_set[0])
    return [make_stream(block_size, frame_length, p) for p in pattern_set]

def make_labeled_sample(context, label, block_size, frame_length, pattern_set):
    """Generate a tuple consisting of the given context and label and a sample
    generated with the given parameters"""
    return LabeledSample(context, label,
                         make_sample(block_size, frame_length, pattern_set))

def make_random_labeled_pattern_sets(context_specs, num_channels, num_blocks,
                                     min_rate, max_rate):
    """Generate a list of LabeledPatternSet objects, each consisting of a
    context id, a pattern_set generated according to the given parameters,
    and a unique label (0, 1, 2,...) and an associated :param num_labels:
    :param context_specs: list of ContextSpec objects
    :param num_channels: specifies the number of patterns in a set
    :param num_blocks: the number of blocks in a pattern
    :param min_rate: minimum avgfiring rate in a block
    :param max_rate: maximum avgfiring rate in a block
    """
    # Create a random pattern set for each 'pattern' index found in context_specs
    #
    pattern_ids = np.unique([csp.pattern_id for csp in context_specs])
    pattern_set_dict = {}
    for pid in pattern_ids:
        pattern_set_dict.update({pid : make_random_pattern_set(num_channels, num_blocks,
                                                               min_rate, max_rate)})
    # Create a list of labeled pattern sets according to context_specs
    #
    lps = []
    for csp in context_specs:
        lps.append(LabeledPatternSet(csp.ctx, csp.label,
                                     pattern_set_dict[csp.pattern_id]))
    return lps

def make_random_dataset(context_specs, num_samples_per_csp,
                        block_size, frame_length,
                        num_channels, num_blocks, min_rate, max_rate,
                        generator_type=Generator.POISSON):
    """Generate a dataset containing a randomized list of num_samples labeled samples."""
    # Create a list of LabeledPatternSets corresponding to the ContextSpecs
    #
    labeled_pattern_sets = make_random_labeled_pattern_sets(context_specs,
                                                            num_channels, num_blocks,
                                                            min_rate, max_rate)

    # Generate labeled_samples from labeled_pattern_sets
    #
    labeled_samples = []
    description = ""
    set_generator_type(generator_type)

    for lps in labeled_pattern_sets:
        description += "{} {} {}\n".format(lps.context, lps.label,
                                           ut.deep_fmt("{:4}", lps.pattern_set))

        for i in range(num_samples_per_csp):
            labeled_samples.append(make_labeled_sample(lps.context, lps.label, block_size,
                                                       frame_length, lps.pattern_set))
    np.random.shuffle(labeled_samples)
    return Dataset(description, block_size, labeled_samples, labeled_pattern_sets=labeled_pattern_sets)
        
def make_dataset(block_size, frame_length,
                 labeled_pattern_sets,
                 description,
                 num_samples_per_pattern_set,
                 generator_type = Generator.POISSON,
                 include_pattern_sets=False):
    """Generate dataset using a given list of LabeledPatternSet.
    :param labeled_pattern_sets: A list of LabeledPatternSets.
    :param description: Textual description to store in the dataset.
    :param num_samples_per_pattern_set: How many samples to generate per pattern set.
    :param include_pattern_sets: Include labeled_pattern_sets in the Dataset."""

    # How often to print a trace message to get max 100 traces
    #
    n = len(labeled_pattern_sets)
    trace_intrvl = pow(10, math.ceil(math.log10(n / 100)))

    set_generator_type(generator_type)
    labeled_samples = []
    for i, lps in enumerate(labeled_pattern_sets):
        if i % trace_intrvl == 0:
            ut.info(f'making sample from pattern_set {i}/{n}')
        for i in range(num_samples_per_pattern_set):
            labeled_samples.append(
                make_labeled_sample(lps.context, lps.label, block_size, frame_length,
                                    lps.pattern_set))
    #np.random.shuffle(labeled_samples)
    lps = labeled_pattern_sets if include_pattern_sets else None
    return Dataset(description, block_size, labeled_samples,
                   labeled_pattern_sets=lps)

class Task:
    """A group of datasets for training, testing and validation.
       If only the train_set is present, then the application 
       may use parts of it for testing and/or validation."""
    def __init__(self, description,
                 train_dataset, test_dataset, val_dataset):
        self.description = description
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset

def save_tasks(tasks : list[Task] , fname : str):
    """Save a list of tasks to file.
    :param tasks: the tasks
    :param fname: The filename (path) whither to save
    """
    f = open(fname, 'wb')
    pickle.dump(tasks, f)
    f.close()

def load_tasks(fname):
    """Load a list of tasks from file.
    :param fname: Filename (path) whence to load
    """
    try:
        # Allow gzipped or plain pickle files.
        try:
            f = gzip.open(fname, 'rb')
            tasks = pickle.load(f)
        except OSError as exc:
            f = open(fname, 'rb')
            tasks = pickle.load(f)

        f.close()

        # Verify that this is a list of Tasks
        #
        ut.abort_unless(isinstance(tasks, list),
                        f'content of file {fname} is not a list')
        ut.abort_unless(len(tasks) > 0,
                        f'content of file {fname} is an empty list')
        ut.abort_unless(isinstance(tasks[0], Task),
                        f'content of file {fname} is not list of Task')

        return tasks
    except IOError as exc:
        ut.fatal(str(exc))
