"""PEALE dataset module."""

import os

import numpy as np
from scipy.misc import imread, imsave

import fhd
from fhd import FHD

DATASET_PATH = os.path.join(os.path.dirname(__file__),
                            'datasets/peale/')

EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__),
                                'experiments/peale/')

RGB_TO_LUMA = (0.299, 0.587, 0.114)


def dataset():
    """
    Load all samples from the PEALE dataset.

    Returns
    -------
    samples : list
        The list of all samples from the PEALE dataset.

    """
    samples = []
    for root, dirnames, filenames in os.walk(DATASET_PATH):
        if not filenames:
            continue
        label = int(root[-2:])
        for filename in filenames:
            name = int(os.path.splitext(filename)[0])
            samples.append(Sample(label=label, name=name))
    return samples


def sample(label, name):
    """Load a sample from the PEALE dataset."""
    return Sample(label, name)


def _get_params(path):
    """Return a dict containing the parameters of an experiment."""
    relpath = os.path.basename(os.path.normpath(path))
    p = relpath.split('-')
    params = {'N': int(p[0]), 'num_dirs': int(p[1]),
              'shape_force': float(p[2]), 'spatial_force': float(p[3]),
              'spatial_radius': int(p[4]), 'range_radius': float(p[5]),
              'min_density': int(p[6])}
    return params


class Sample(object):

    """
    A sample is a butterfly image in the PEALE dataset.

    Parameters
    ----------
    label : int
        Label of the sample.
    name : int
        Name of the sample.

    Attributes
    ----------
    im : array_like
        Initial butterfly image.
    label : int
        Label of the sample.
    name : int
        Name of the sample.
    meanshift : array_like
        Butterfly segmented by meanshift.
    kmeans : array_like
        Butterfly segmented by kmeans.
    layers : array_like
        Binary images computed from `kmeans`.
    fhd : FHD
        FHD descriptor computed from `layers`.

    Raises
    ------
    ValueError
        If `label` and `name` don't correspond to an existing sample.

    """

    def __init__(self, label, name, path=None):
        try:
            self.im = self._imread(label, name)
        except FileNotFoundError:
            raise ValueError('Invalid PEALE sample.')
        self.label = label
        self.name = name
        if path:
            params = _get_params(path)
            path = os.path.join(path, str(label).zfill(2), str(name).zfill(2))
            self.meanshift = imread(os.path.join(path, 'meanshift.png'))
            self.kmeans = imread(os.path.join(path, 'kmeans.png'))
            self.layers = []
            for i in range(params['N']):
                self.layers.append(
                    imread(os.path.join(path, 'layers-{}.png'.format(i))))
            self.fhd = FHD.load(os.path.join(path, 'fhd.txt'), params['N'],
                                params['shape_force'], params['spatial_force'])

    def _imread(self, label, name):
        """Return the butterfly image of the requested sample."""
        label = str(label).zfill(2)
        name = str(name).zfill(2) + '.png'
        return imread(os.path.join(DATASET_PATH, label, name))

    def segment(self, num_clusters, spatial_radius, range_radius, min_density):
        """Segment the butterfly image of the current sample."""
        # Meanshift
        segm, num_modes = fhd.meanshift(self.im, spatial_radius,
                                        range_radius, min_density)
        self.meanshift, self.num_modes = segm, num_modes
        # KMeans
        segm, clusters = fhd.kmeans(self.meanshift, num_clusters)
        self.kmeans = segm
        # Binary layers
        self.clusters = np.array(
            sorted(clusters, key=lambda c: c.dot(RGB_TO_LUMA), reverse=True))
        self.layers = fhd.layers(self.kmeans, self.clusters)

    def compute_fhd(self, num_dirs, shape_force, spatial_force):
        """Compute FHD descriptor of the current samples."""
        self.fhd = fhd.fhd(self.layers, num_dirs, shape_force, spatial_force)

    def dump(self, path):
        """Dump the object in directory structure starting with base path."""
        path = os.path.join(
            path, str(self.label).zfill(2), str(self.name).zfill(2))
        if not os.path.exists(path):
            os.makedirs(path)
        meanshift_path = 'meanshift.png'
        kmeans_path = 'kmeans.png'
        imsave(os.path.join(path, meanshift_path), self.meanshift)
        imsave(os.path.join(path, kmeans_path), self.kmeans)
        for index, layer in enumerate(self.layers):
            layer_path = 'layers-{}.png'.format(index)
            imsave(os.path.join(path, layer_path), layer)
        self.fhd.dump(os.path.join(path, 'fhd.txt'))


class Experiment(object):

    """An experiment on the PEALE dataset."""

    def __init__(self, path=None, normalized=False):
        if not path:
            self.samples = dataset()
        else:
            self.samples = []
            for str_label in os.listdir(path):
                label = int(str_label)
                label_path = os.path.join(path, str_label)
                for str_name in os.listdir(label_path):
                    name = int(str_name)
                    sample = Sample(label, name, path)
                    if normalized:
                        sample.fhd.normalize()
                    self.samples.append(sample)
        self.num_samples = len(self.samples)

    def cross_validate(self, metric='L2', matching='default', alpha=None):
        """Leave-one-out cross validation."""
        from sklearn.cross_validation import LeaveOneOut
        loo = LeaveOneOut(self.num_samples)
        for train, test in loo:
            A = self.samples[test[0]]
            A.neighbors = [self.samples[i] for i in train]
            A.neighbors.sort(key=lambda B: fhd.distance(A.fhd, B.fhd, metric,
                                                        matching, alpha))
            print('[{}/{}] label={}, nearest_neighbor={}'.format(
                str(test[0] + 1).zfill(len(str(self.num_samples))),
                self.num_samples, str(A.label).zfill(2),
                str(A.neighbors[0].label).zfill(2)))

    def recognition_rates(self):
        labels = {}
        true_positives = {}
        for sample in self.samples:
            if sample.label not in labels:
                labels[sample.label] = 0
                true_positives[sample.label] = 0
            labels[sample.label] += 1
            if sample.label == sample.neighbors[0].label:
                true_positives[sample.label] += 1
        for label in labels:
            print('label {}: {}/{}, {}%'.format(
                str(label).zfill(2),
                str(true_positives[label]).zfill(2),
                str(labels[label]).zfill(2),
                round((true_positives[label] / labels[label]) * 100, 2)))
        total_tp = 0
        for label in true_positives:
            total_tp += true_positives[label]
        print('Recognition rate: {}%'.format(
            round((total_tp / self.num_samples) * 100, 2)))
        mean_recognition_rate = 0
        for label in true_positives:
            mean_recognition_rate += \
                (true_positives[label] / labels[label]) * 100
        mean_recognition_rate /= 28
        print('Mean recognition rate: {}%'.format(
            round(mean_recognition_rate, 2)))

    def __getitem__(self, index):
        """Return a sample of the Peale experiment by index."""
        return self.samples[index]

    def run_experiment(self, N, num_dirs, shape_force, spatial_force,
                       spatial_radius, range_radius, min_density):
        """Segmentation and FHD descriptors for the Peale dataset."""
        path = '-'.join((str(N), str(num_dirs), str(shape_force),
                         str(spatial_force), str(spatial_radius),
                         str(range_radius), str(min_density)))
        path = os.path.join(EXPERIMENTS_PATH, path)
        for index, sample in enumerate(self.samples):
            print('[{}/{}] label={}, name={}'.format(
                str(index + 1).zfill(len(str(self.num_samples))),
                self.num_samples, str(sample.label).zfill(2),
                str(sample.name).zfill(2)))
            sample.segment(N, spatial_radius, range_radius, min_density)
            sample.compute_fhd(num_dirs, shape_force, spatial_force)
            sample.dump(path)
