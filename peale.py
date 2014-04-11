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
        self.clusters = np.array(
            sorted(clusters, key=lambda c: c.dot([0.299, 0.587, 0.114])))
        self.split_into_layers()

    def split_into_layers(self):
        """Split the the current sample into binary layers."""
        self.layers = fhd.binary_layers(self.kmeans, self.clusters)

    def compute_fhd(self, num_dirs, shape_force, spatial_force):
        """Compute FHD descriptor of the current samples."""
        self.fhd = FHD.compute_fhd(self.layers, num_dirs, shape_force,
                                   spatial_force)

    def dump(self, base_path):
        """Dump the object in directory structure starting with base path."""
        path = os.path.join(base_path, self.str_label(), self.str_label())
        if not os.path.exists(path):
            os.makedirs(path)
        meanshift_path = '01-meanshift-{}.png'.format(self.num_modes)
        kmeans_path = '02-kmeans-{}.png'.format(self.clusters.shape[0])
        imsave(os.path.join(path, meanshift_path), self.meanshift)
        imsave(os.path.join(path, kmeans_path), self.kmeans)
        for index, layer in enumerate(self.layers):
            layer_path = 'layer-{}.png'.format(index)
            imsave(os.path.join(path, layer_path), layer)

    def str_label(self):
        """Return string version of label attribute."""
        return str(self.label).zfill(2)

    def str_name(self):
        """Return string version of name attribute."""
        return str(self.name).zfill(2) + '.jpg'

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
