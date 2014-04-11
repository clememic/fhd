"""PEALE dataset module."""

import os

import numpy as np
from scipy.misc import imread, imsave

import fhd
from fhd import FHD


class Peale(object):

    """A Peale object is an sample (a butterfly) of the PEALE dataset."""

    DATASET_PATH = os.path.join(os.path.dirname(__file__), 'datasets/peale/')

    def __init__(self, label, name):
        """Load a sample from the PEALE dataset."""
        self.label = label
        self.name = name
        self.image = self.imread()

    def imread(self):
        """Read and return the butterfly image of the current sample."""
        dataset_path = self.__class__.DATASET_PATH
        label = self.str_label()
        name = self.str_name()
        image = imread(os.path.join(dataset_path, label, name))
        return image


    def segment(self, num_clusters, spatial_radius, range_radius, min_density):
        """Segment the butterfly image of the current sample."""
        segm, num_modes = fhd.meanshift(self.image, spatial_radius,
                                        range_radius, min_density)
        bg = (segm == segm[0, 0]).all(segm.ndim - 1) # background mask
        segm[bg] = np.zeros(segm.shape[-1]) # background in black
        num_modes -1 # background doesn't count
        self.meanshift = segm.copy()
        self.num_modes = num_modes
        segm[~bg], clusters = fhd.kmeans(segm[~bg], num_clusters)
        self.kmeans = segm
        self.clusters = clusters

    def split_into_layers(self):
        """Split the the current sample into binary layers."""
        self.layers = fhd.binary_layers(self.kmeans, self.clusters)

    def compute_fhd(self, num_dirs, shape_force, spatial_force):
        """Compute FHD descriptor of the current samples."""
        self.fhd = FHD.compute_fhd(self.layers, num_dirs, shape_force,
                                   spatial_force)

    def str_label(self):
        """Return string version of label attribute."""
        return str(self.label).zfill(2)

    def str_name(self):
        """Return string version of name attribute."""
        return str(self.name).zfill(2) + '.png'

    @classmethod
    def dataset(cls):
        """Return a list of all samples from the PEALE dataset."""
        peales = []
        for root, dirnames, filenames in os.walk(cls.DATASET_PATH):
            if not filenames:
                continue
            label = int(root[-2:])
            for filename in filenames:
                name = int(os.path.splitext(filename)[0])
                peales.append(cls(label=label, name=name))
        return peales
