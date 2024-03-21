#!/usr/bin/env python
# coding: utf-8

# Convert (a subset of) the MNIST or FMNIST dataset to a directory of
# image files suitable for the CustomImageDataset class described in
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#
# All the training images are deposited in a flat directory, together with a
# CSV file that maps image file names to numerical labels, and similarly for
# the test images:
#
# root/train/train_0.jpeg
# ...
# root/train/train_59999.jpeg
# root/train/train_labels.csv
# root/train/
# root/test/test_0.jpeg
# ...
# root/test/test_9999.jpeg
# root/test/test_labels.csv

# Imports
#
from __future__ import print_function, division
import os
import sys
import numpy as np
from mnist.loader import MNIST
from pathlib import Path
from PIL import Image

# Local imports
scriptpath = "../lib/"
sys.path.append(os.path.abspath(scriptpath))

import util as ut

ut.set_trace_level(ut.Trace.INFO)

# Hyperparameters
# Note: these can be changed on the command line
#
hp = {}

def parse_cmdline():
    parser = ut.TerseArgumentParser(description='Make torchvision.ImageFolder')

    parser.add_str_arg('source_dir', '/home/peter/win_peter/work/PycharmProjects/data/mnist',
                       'Directory containing MNIST or FMNIST dataset')
    parser.add_str_arg('output_dir', './data2/mnist', 'Root of output directory tree')
    parser.add_int_arg('num_train_samples', None,   'Number of training samples (None means all)')
    parser.add_int_arg('num_test_samples',  None,   'Number of test samples (None means all)')
    global hp
    hp = parser.parse_args()
            
# A partial MNIST or FMNIST Dataset containing labeled train and test data
#
class Dataset:
    def __init__(self, train_images, train_labels, test_images, test_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        self.test_images = test_images
        self.test_labels = test_labels

# Get a specified MNIST or FMNIST sub-dataset
#
def get_dataset(num_train_samples, num_test_samples):
    """Build a set of the num_train_samples first train images and
    num_test_samples first test images in the MNIST data set.
    Params:
        num_train_samples: Number of training samples
        num_test_samples: Number of test samples
    Return:
        Dataset object"""

    # Read the whole MNIST or FMNIST dataset
    #
    mnist_data = MNIST(hp.source_dir)
    all_train_images, all_train_labels = mnist_data.load_training()
    all_test_images, all_test_labels = mnist_data.load_testing()

    # Extract subsets of MNIST or FMNIST
    #
    train_images = np.asarray(all_train_images[:num_train_samples]).astype('float64')
    train_labels = all_train_labels[:num_train_samples]
    test_images = np.asarray(all_test_images[:num_test_samples]).astype('float64')
    test_labels = all_test_labels[:num_test_samples]

    return Dataset(train_images, train_labels, test_images, test_labels)

def main():
    parse_cmdline()

    # Print out the hyperparam values
    # (Very useful when saving the output to a file)
    #
    ut.print_dict(vars(hp))

    # Load train and test data
    #
    ut.tprint('Load data')
    dataset = get_dataset(hp.num_train_samples, hp.num_test_samples)
    ut.tprint('data loaded')

    for (subdir, labels, images) in (('train', dataset.train_labels, dataset.train_images),
                                     ('test', dataset.test_labels, dataset.test_images)):

        dirpath = Path(hp.output_dir + '/' + subdir + '/')
        dirpath.mkdir(parents=True, exist_ok=True)
        dirstr = '/'.join((dirpath.parts))

        img_label_map = {}

        for i, (label, img) in enumerate(zip(labels, images)):
            imgname = '.'.join((subdir + '_' + str(i), 'jpg'))
            imgpath = '/'.join((dirstr, imgname))

            im = Image.fromarray(img.reshape((28, 28)))
            im = im.convert("L")
            im.save(imgpath)
            img_label_map[imgname] = label

        csv_fname = '/'.join((dirstr, subdir + '_' + 'labels.csv'))
        ut.dict_to_csv_file(img_label_map, csv_fname)

if __name__ == '__main__':
    main()
