"""FHD descriptors module."""

import math
import os

import numpy as np

import hdist

_lib_fh = os.path.join(os.path.dirname(__file__), 'libfhistograms_raster.so')

matchings = ['default', 'greedy']


def fhistogram(A, B=None, num_dirs=180, force_type=0.0):
    """
    Compute an FHistogram between two binary images A and B.

    A and B must be two binary images of the same shape. If only A is provided,
    the FHistogram is computed with itself. num_dirs (> 0) is the number of
    directions to consider. force_type is the value of the attraction force.
    """
    if B is None:
        B = A
    if A.shape != B.shape:
        raise ValueError('A and B must have the same shape.')
    num_dirs = int(num_dirs)
    if num_dirs <= 0:
        raise ValueError('num_dirs must be > 0.')
    if force_type < 0:
        raise ValueError('force_type must be >= 0.')
    if B is A and force_type >= 1:
        raise ValueError('0 <= force_type < 1 when B == A.')
    # Load C shared library
    import ctypes
    clib = ctypes.cdll.LoadLibrary(_lib_fh)
    # Compute FHistogram between A and B
    fhistogram = np.ndarray(num_dirs)
    height, width = A.shape
    clib.FRHistogram_CrispRaster(
        fhistogram.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(num_dirs),
        ctypes.c_double(force_type),
        A.ctypes.data_as(ctypes.c_char_p),
        B.ctypes.data_as(ctypes.c_char_p),
        ctypes.c_int(width),
        ctypes.c_int(height))
    return fhistogram


def distance(A, B, metric='L2', matching='default', alpha=None):

    """
    Distance between two FHD descriptors.

    alpha must be between 0 and 1 and is a weight level given to the distance
    between shapes (comparison of N FHistograms) compared to the distance
    between spatial relations (comparison of (N * (N - 1) / 2) FHistograms).
    By default, alpha is computed so that shape and spatial relations have the
    same weight no matter how many FHistograms they contain.
    """

    if A.N != B.N:
        raise ValueError('A and B should have the same size.')
    N = A.N
    if alpha is None:
        alpha = 1 - (2 / (N + 1))
    elif not 0 <= alpha <= 1:
        raise ValueError('alpha should be between 0 and 1.')
    if matching not in matchings:
        raise ValueError('Incorrect matching strategy.')

    if matching == 'default':
        shape_distance = 0.0
        spatial_distance = 0.0
        for i in range(N):
            shape_distance += hdist.distance(A[i, i], B[i, i], metric)
            for j in range(i + 1, N):
                spatial_distance += hdist.distance(A[i, j], B[i, j], metric)

    elif matching == 'greedy':
        # First compute distance between shapes by matching them
        matching_AB = greedy_shape_matching(A, B, metric)
        matching_BA = greedy_shape_matching(B, A, metric)
        if matching_AB[0] <= matching_BA[0]:
            shape_distance = matching_AB[0]
            matching = matching_AB[1]
        else:
            shape_distance = matching_BA[0]
            matching = matching_BA[1]
        # Then distance between spatial relations based on matching (matrix
        # must be reorganized)
        spatial_distance = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                mi, mj = matching[i], matching[j]
                if mi <= mj:
                    spatial_distance += hdist.distance(
                        A[i, j], B[mi, mj], metric)
                else:
                    # In this case, the FHistogram in B must be shifted by
                    # half its size to mimic the lower diagonal of the matrix
                    mi, mj = mj, mi
                    pivot = B.num_dirs // 2
                    spatial_distance += hdist.distance(
                        A[i, j], np.roll(B[mi, mj], pivot), metric)

    return (alpha * shape_distance) + ((1 - alpha) * spatial_distance)


def greedy_shape_matching(A, B, metric='L2'):
    """
    Return a greedy shape matching between A and B and the associated distance.

    The matching is based on shape information and is computed in a greedy way,
    that is, each shape of A is matched with its closest shape in B. Such a
    strategy doesn't guarantee an optimal matching, and most importantly
    doesn't yeild a symmetric distance (a greedy matching from A to B can be
    different than from B to A). Symmetry can be preserved by computing both
    matchings (AB and BA) and keeping the mininmum.
    """
    if A.N != B.N:
        raise ValueError('A and B should have the same size.')
    N = A.N
    distance = 0.0
    matching = {}
    choices = [n for n in range(N)]
    for i in range(N):
        dists = {}
        for j in choices:
            dists[hdist.distance(A[i, i], B[j, j], metric)] = j
        min_dist = min(dists)
        distance += min_dist
        matching[i] = dists[min_dist]
        choices.remove(dists[min_dist])
    return distance, matching


def optimal_shape_matching(A, B, metric='L2'):
    """
    Return an optimal shape matching between two FHD descriptors, and the
    resulting sum of distances between shapes.

    This function computes the distance between all possible permutations of
    shapes of `A` and `B` and returns the minimum one. Careful, it doesn't
    scale: complexity is N! where N is the number of layers.

    Parameters
    ----------
    A : FHD object
        FHD descriptor, must have the same size as `B`.
    B : FHD object
        FHD Descriptor, must have the same size as `A`.
    metric : {'L1', 'L2', 'CHI2'}, optional
        Metric used to compute the matching. Default is 'L2'.

    Returns
    -------
    distance : float
        The optimal distance between shapes of `A` and `B`.
    matching : dict
        The optimal matching between shapes of `A` and `B`.

    Raises
    ------
    ValueError
        If `A` and `B` don't have the same size.

    """
    if A.N != B.N:
        raise ValueError('A and B should have the same size.')
    N = A.N
    # Compute distances between each shape of A and B (N * N matrix)
    distances = np.ndarray((N, N))
    for i in range(N):
        for j in range(N):
            distances[i, j] = hdist.distance(A[i, i], B[j, j], metric)
    # Compute sum of distances for all possible permutations
    matchings = {}
    from itertools import permutations
    for permutation in permutations(range(N)):
        distance = 0.0
        for i, j in enumerate(permutation):
            distance += distances[i, j]
        matchings[distance] = permutation
    # Return the best matching
    min_distance = min(matchings.keys())
    best_permutation = matchings[min_distance]
    optimal_matching = {}
    for i in range(N):
        optimal_matching[i] = best_permutation[i]
    return min_distance, optimal_matching


class FHD(object):

    """FHistogram Decomposition descriptor."""

    def __init__(self, fhistograms, shape_force, spatial_force):
        """Create an FHD descriptor."""
        self.N = fhistograms.shape[0]
        self.num_dirs = fhistograms.shape[2]
        self.fhistograms = fhistograms
        self.shape_force = shape_force
        self.spatial_force = spatial_force

    def __getitem__(self, index):
        """Return FHistogram by index."""
        return self.fhistograms[index]

    def normalize(self):
        for i in range(self.N):
            for j in range(i, self.N):
                self.fhistograms[i, j] /= self.fhistograms[i, j].max()

    def dump(self, filename):
        """Dump FHD descriptor to file."""
        np.savetxt(filename, self.fhistograms[np.triu_indices(self.N)])

    @classmethod
    def load(cls, filename, N, shape_force, spatial_force):
        fhistograms_from_file = np.loadtxt(filename)
        num_dirs = fhistograms_from_file.shape[-1]
        fhistograms = np.ndarray((N, N, num_dirs))
        fhistograms[np.triu_indices(N)] = fhistograms_from_file
        return cls(fhistograms, shape_force, spatial_force)

    @classmethod
    def compute_fhd(cls, layers, num_dirs=180, shape_force=0.0,
                    spatial_force=0.0):
        """Compute an FHD descriptor for given layers."""
        N = len(layers)
        fhistograms = np.ndarray((N, N, num_dirs))
        for i in range(N):
            for j in range(i, N):
                fhistograms[i, j] = fhistogram(
                    layers[i], layers[j], num_dirs,
                    shape_force if i == j else spatial_force)
        return cls(fhistograms, shape_force, spatial_force)


def meanshift(image, spatial_radius, range_radius, min_density):
    """Segment an image with meanshift."""
    import pymeanshift as pyms
    segm, labels, num_modes = pyms.segment(image, spatial_radius, range_radius,
                                           min_density, pyms.SPEEDUP_MEDIUM)
    return segm, num_modes


def kmeans(samples, num_clusters):
    """Perform kmeans clustering on given samples."""
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(samples)
    clusters = kmeans.cluster_centers_.astype(np.uint8)
    labels = kmeans.labels_
    for index in range(samples.shape[0]):
        samples[index] = clusters[labels[index]]
    return samples, clusters


def binary_layers(segm, clusters):
    """Split a segmented image into binary layers."""
    N = clusters.shape[0]
    layers = [np.zeros((segm.shape[0], segm.shape[1]), np.uint8)
              for i in range(N)]
    from scipy.ndimage import binary_erosion
    for index, cluster in enumerate(clusters):
        layers[index][np.where((segm == cluster).all(segm.ndim - 1))] = 255
        layers[index] = binary_erosion(layers[index], np.ones((3, 3)))
    return layers
