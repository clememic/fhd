"""Datasets module."""

import os

import numpy as np
from skimage.io import ImageCollection
from sklearn.datasets.base import Bunch

PATH = os.path.join(os.path.dirname(__file__), 'datasets')


def load_peale():
    """Load the PEALE dataset."""
    images = ImageCollection(os.path.join(PATH, 'peale/*.png'))
    labels = np.array([int(f.split('/')[-1].split('-')[0])
                       for f in images.files])
    return Bunch(name='peale', images=images, labels=labels)


def load_coil():
    """Load the Coil-100 dataset."""
    images = ImageCollection(os.path.join(PATH, 'coil/*.png'))
    labels = np.array([int(f.split('obj')[-1].split('__')[0])
                       for f in images.files])
    return Bunch(name='coil', images=images, labels=labels)


def load_motos():
    """"Load the Motos dataset."""
    images = ImageCollection(os.path.join(PATH, 'motos/*.jpg'))
    return Bunch(name='motos', images=images)


def load_motos_labeled():
    """Load the labeled Motos dataset."""
    images = ImageCollection(os.path.join(PATH, 'motos-labeled/*.jpg'))
    labels = np.array([int(f.split('/')[-1].split('-')[0])
                       for f in images.files])
    return Bunch(name='motos-labeled', images=images, labels=labels)


def load(dataset):
    """Load a dataset by name."""
    if dataset == 'peale':
        dataset = load_peale()
    elif dataset == 'coil':
        dataset = load_coil()
    elif dataset == 'motos':
        dataset = load_motos()
    elif dataset == 'motos-labeled':
        dataset = load_motos_labeled()
    else:
        raise ValueError('Incorrect dataset.')
    return dataset
