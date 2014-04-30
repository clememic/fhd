"""PEALE dataset module."""

import os

import numpy as np
from scipy.misc import imread, imsave

import fhd
from fhd import FHD

DATASET_PATH = os.path.join(os.path.dirname(__file__), 'datasets/peale/')

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


class Sample(object):

    """
    A sample is a butterfly image in the PEALE dataset.

    Parameters
    ----------
    label : int
        The label of the sample.
    name : int
        The name of the sample.

    Attributes
    ----------

    Notes
    -----

    """

    def __init__(self, label, name, path=None):
        self.label = label
        self.name = name
        self.image = self._imread()
        if path:
            params = PealeExperiment.get_params(path)
            path = os.path.join(path, str(label).zfill(2), str(name).zfill(2))
            self.meanshift = imread(os.path.join(path, 'meanshift.png'))
            self.kmeans = imread(os.path.join(path, 'kmeans.png'))
            self.layers = []
            for i in range(params['N']):
                self.layers.append(
                    imread(os.path.join(path, 'layers-{}.png'.format(i))))
            self.fhd = FHD.load(os.path.join(path, 'fhd.txt'), params['N'],
                                params['shape_force'], params['spatial_force'])

    def _imread(self):
        """Return the butterfly image of the current sample."""
        label = str(self.label).zfill(2)
        name = str(self.name).zfill(2) + '.jpg'
        return imread(os.path.join(DATASET_PATH, label, name))

    def segment(self, num_clusters, spatial_radius, range_radius, min_density):
        """Segment the butterfly image of the current sample."""
        segm, num_modes = fhd.meanshift(self.image, spatial_radius,
                                        range_radius, min_density)
        bg = (segm == segm[0, 0]).all(segm.ndim - 1)  # background mask
        segm[bg] = np.zeros(segm.shape[-1])  # background in black
        num_modes-1  # background doesn't count
        self.meanshift = segm.copy()
        self.num_modes = num_modes
        segm[~bg], clusters = fhd.kmeans(segm[~bg], num_clusters)
        self.kmeans = segm
        self.clusters = np.array(
            sorted(clusters, key=lambda c: c.dot(RGB_TO_LUMA)))
        self.split_into_layers()

    def split_into_layers(self):
        """Split the the current sample into binary layers."""
        self.layers = fhd.binary_layers(self.kmeans, self.clusters)

    def compute_fhd(self, num_dirs, shape_force, spatial_force):
        """Compute FHD descriptor of the current samples."""
        self.fhd = FHD.compute_fhd(self.layers, num_dirs, shape_force,
                                   spatial_force)

    def dump(self, path):
        """Dump the object in directory structure starting with base path."""
        path = os.path.join(
            path, str(self.label).zfill(2), str(self.name).zfill(2))
        if not os.path.exists(path):
            os.makedirs(path)
        meanshift_path = 'meanshift.png'.format(self.num_modes)
        kmeans_path = 'kmeans.png'.format(self.clusters.shape[0])
        imsave(os.path.join(path, meanshift_path), self.meanshift)
        imsave(os.path.join(path, kmeans_path), self.kmeans)
        for index, layer in enumerate(self.layers):
            layer_path = 'layers-{}.png'.format(index)
            imsave(os.path.join(path, layer_path), layer)
        self.fhd.dump(os.path.join(path, 'fhd.txt'))


class PealeExperiment(object):

    """Represent an "FHD experiment" on the Peale dataset."""

    EXPERIMENTS_BASE_PATH = os.path.join(os.path.dirname(__file__),
                                         'experiments/peale/')

    def __init__(self, experiment_path=None):
        """Initialize a Peale experiment."""
        if not experiment_path:
            self.samples = Peale.dataset()
        else:
            self.samples = []
            params = self.__class__.get_params(experiment_path)
            for str_label in os.listdir(experiment_path):
                label = int(str_label)
                label_path = os.path.join(experiment_path, str_label)
                for str_name in os.listdir(label_path):
                    name = int(str_name)
                    self.samples.append(Peale(label, name, experiment_path))
        self.num_samples = len(self.samples)

    def cross_validate(self, metric='L2', matching='default', alpha=None):
        """Leave-one-out cross validation."""
        from sklearn.cross_validation import LeaveOneOut
        loo = LeaveOneOut(self.num_samples)
        for train, test in loo:
            A = self.samples[test[0]]
            print('[{}/{}] label={}, name={}'.format(
                str(test[0] + 1).zfill(len(str(self.num_samples))),
                self.num_samples, A.str_label(), A.str_name()))
            A.neighbors = [self.samples[i] for i in train]
            A.neighbors.sort(key=lambda B: fhd.distance(A.fhd, B.fhd, metric,
                                                        matching, alpha))

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
            mean_recognition_rate += (true_positives[label] / labels[label]) * 100
        mean_recognition_rate /= 28
        print('Mean recognition rate: {}%'.format(
            round(mean_recognition_rate, 2)))

    def __getitem__(self, index):
        """Return a sample of the Peale experiment by index."""
        return self.samples[index]

    def run_experiment(self, N, num_dirs, shape_force, spatial_force,
                       spatial_radius, range_radius, min_density):
        """Segmentation and FHD descriptors for the Peale dataset."""
        experiment_path = '-'.join((str(N), str(num_dirs), str(shape_force),
                                   str(spatial_force), str(spatial_radius),
                                   str(range_radius), str(min_density)))
        base_path = os.path.join(self.__class__.EXPERIMENTS_BASE_PATH,
                                 experiment_path)
        for index, sample in enumerate(self.samples):
            print('[{}/{}] label={}, name={}'.format(
                str(index + 1).zfill(len(str(self.num_samples))),
                self.num_samples, sample.str_label(), sample.str_name()))
            sample.segment(N, spatial_radius, range_radius, min_density)
            sample.compute_fhd(num_dirs, shape_force, spatial_force)
            sample.dump(base_path)

    @classmethod
    def get_params(cls, experiment_path):
        """Return a dict containing the parameters of an experiment."""
        relpath = os.path.relpath(experiment_path, cls.EXPERIMENTS_BASE_PATH)
        p = relpath.split('-')
        params = {'N': int(p[0]), 'num_dirs': int(p[1]),
                  'shape_force': float(p[2]), 'spatial_force': float(p[3]),
                  'spatial_radius': int(p[4]), 'range_radius': float(p[5]),
                  'min_density': int(p[6])}
        return params
