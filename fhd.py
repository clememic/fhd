"""
FHD descriptors module.
"""

import ctypes
import os

import numpy as np
import pymeanshift as pyms
from scipy.ndimage import binary_erosion
from sklearn.cluster import KMeans

import hdist

# Load C shared library for FHistograms
libfh = os.path.join(os.path.dirname(__file__), 'libfhistograms_raster.so')
clib = ctypes.cdll.LoadLibrary(libfh)

# Different matching strategies
matchings = ('default', 'greedy', 'optimal')

# Conversion from RGB to Luma value
rgb_to_luma = (0.299, 0.587, 0.114)


def meanshift(image, spatial_radius, range_radius, min_density):
    """
    Segment an image using the meanshift clustering algorithm.

    Arguments
    ---------
    image : array_like
        Input image.
    spatial_radius : int
        Spatial radius of the search window.
    range_radius : float
        Range radius parameter of the search window.
    min_density : int
        Minimum size of a region in the segmented image.

    Returns
    -------
    segmented : array_like
        Segmented image.
    num_modes : int
        The number of modes found by the meanshift algorithm.

    Notes
    -----
    A custom fork of the "pymeanshift" module [1] is used to perform the
    meanshift. Details on the algorithm can be found in [2].

    References
    ----------
    [1] https://github.com/clememic/pymeanshift

    [2] Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
        feature space analysis". IEEE Transactions on Pattern Analysis and
        Machine Intelligence. 2002. pp. 603-619.

    """
    hs, hr, M = spatial_radius, range_radius, min_density
    segmented, labels, num_modes = pyms.segment(image, hs, hr, M)
    return segmented, num_modes


def kmeans(image, num_clusters, filter_background=True):
    """
    Segment an image using the kmeans clustering algorithm.

    Arguments
    ---------
    image : array_like
        Input image.
    num_clusters : int
        The number of clusters to form.
    filter_background : bool, optional, default: True
        Wheter the background of the image should be filtered.

    Returns
    -------
    segmented : array_like
        Segmented image.
    clusters : array_like, shape(num_clusters, n_features)
        The formed clusters.

    Notes
    -----
    If `filter_background` is True, this function will form `num_clusters` + 1
    but the cluster containing the top-left pixel will be deleted.
    This function uses the kmeans implementation of the scikit-learn library.
    The centroid seeds are initialized 10 times with the 'kmeans++' method.
    The random initializations are always done with the same random number
    generator so that for a given image, we always get the same clusters.
    The result will be the best output in terms of inertia.

    """
    segmented = image.copy()
    if segmented.ndim == 2:
        samples = segmented.reshape(-1, 1)
    elif segmented.ndim == 3:
        samples = segmented.reshape(-1, 3)
    if filter_background:
        num_clusters += 1
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(samples)
    clusters = kmeans.cluster_centers_.astype(np.uint8)
    for index in range(samples.shape[0]):
        samples[index] = clusters[labels[index]]
    if filter_background:
        clusters = clusters[~(clusters == segmented[0, 0]).all(-1)]
    return segmented, clusters


def layers(segmented, clusters):
    """
    Split a segmented image into binary layers, according to its clusters.

    Arguments
    ---------
    segmented : array_like
        Input segmented image.
    clusters : array_like
        Clusters in the segmented image.

    Returns
    -------
    layers : list of binary images
        Binary layers of the segmented image.

    Notes
    -----
    The clusters are sorted by decreasing luma.
    The layers formed are binary eroded by a 3 * 3 structuring element.

    """
    N = clusters.shape[0]
    clusters = sorted(clusters, key=lambda c: c.dot(rgb_to_luma), reverse=True)
    layers = [np.zeros(segmented.shape[:2], np.uint8) for i in range(N)]
    for index, cluster in enumerate(clusters):
        mask = np.where((segmented == cluster).all(-1))
        layers[index][mask] = 255
        layers[index] = binary_erosion(layers[index], np.ones((3, 3)))
    return layers


def fhistogram(a, b=None, num_dirs=180, force_type=0.0):
    """
    Compute an FHistogram between two binary images.

    The FHistogram is computed between `a` and `b` images, along `num_dirs`
    directions and using the attraction force `force_type`.

    Parameters
    ----------
    a, b : (w, h,) array_like
        The two binary images to consider. `a` and `b` must have the same size.
        If `b` is None, the FHistogram is computer between `a` and `a`.
    num_dirs : int
        The number of directions to consider. Default is 180.
    force_type : float
        The attraction force used to compute the FHistogram. Default is 0.0.

    Returns
    -------
    fh : (num_dirs,) ndarray
        The FHistogram between `a` and `b` along `num_dirs` directions using
        the attraction force `force_type`.

    Notes
    -----
    The FHistogram is computed using a C shared library called with ctypes.

    fhistogram(a, b) represents the spatial position of `a` relative to `b`.
    fhistogram(a, a) is the FHistogram representing the shape of `a`.
    fhistogram(a) is equivalent to fhistogram(a, a).

    The attraction force `force_type` must be < 1 when images are overlapping.

    """
    if b is None:
        b = a
    if a.shape != b.shape:
        raise ValueError('a and b must have the same shape.')
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError('a and b must be 2D with one channel.')
    if b is a and force_type >= 1:
        raise ValueError('0 <= force_type < 1 when b == a.')
    # Compute and return the FHistogram between a and b
    fhistogram = np.ndarray(num_dirs)
    height, width = a.shape
    clib.FRHistogram_CrispRaster(
        fhistogram.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(num_dirs),
        ctypes.c_double(force_type),
        a.ctypes.data_as(ctypes.c_char_p),
        b.ctypes.data_as(ctypes.c_char_p),
        ctypes.c_int(width),
        ctypes.c_int(height))
    return fhistogram


def fhd(layers, num_dirs=180, shape_force=0.0, spatial_force=0.0):
    """
    Compute an FHD descriptor with layers extracted from a segmented image.

    Arguments
    ---------
    layers : list of binary images
        Binary layers extracted from a segmented image.
    num_dirs : int, optional, default=180
        Number of directions to consider for each FHistogram.
    shape_force : float, optional, default=0.0
        Force used for shape FHistograms (diagonal).
    spatial_force : float, optional, default=0.0
        Force used for spatial relations FHistograms (upper triangle).

    Returns
    -------
    The FHD descriptor object.

    """
    N = len(layers)
    fhistograms = np.ndarray((N, N, num_dirs))
    for i in range(N):
        for j in range(i, N):
            fhistograms[i, j] = fhistogram(
                layers[i], layers[j], num_dirs,
                shape_force if i == j else spatial_force)
    return FHD(fhistograms)


class FHD(object):

    """
    FHistogram Decomposition descriptor.

    An FHD object is basically a container for an upper triangular matrix of
    FHistograms.

    Parameters
    ----------
    fhistograms : array_like
        Upper trianguler matrix of FHistograms

    Attributes
    ----------
    N : int
        The number of layers/shapes in the FHD.
    num_dirs : int
        The number of directions for each FHistogram of the FHD.
    fhistograms : (N, N, num_dirs) array_like
        The underlying FHistograms of the FHD.

    """

    def __init__(self, fhistograms):
        """Create an FHD descriptor."""
        self.N = fhistograms.shape[0]
        self.num_dirs = fhistograms.shape[-1]
        self.fhistograms = fhistograms

    def __getitem__(self, index):
        """
        Return FHistograms of the FHD by index.

        This method delegates indexing and slicing of the FHD to its underlying
        ndarray of FHistograms, supporting NumPy's indexing capabilities. For
        convenience, if a single integer index is provided, the method returns
        the shape FHistogram located on the diagonal of the FHD.

        Parameters
        ----------
        index : int or array_like
            Value by which the FHD is indexed.

        Returns
        -------
        Indexed FHD by its FHistograms.

        """
        try:
            index = int(index)
            return self.fhistograms[index, index]
        except TypeError:
            return self.fhistograms[index]

    def __iter__(self):
        return iter(self.fhistograms)

    def normalize(self):
        for i in range(self.N):
            for j in range(i, self.N):
                self.fhistograms[i, j] /= self.fhistograms[i, j].max()

    def dump(self, filename):
        """Dump FHD descriptor to file."""
        np.savetxt(filename, self.fhistograms[np.triu_indices(self.N)])

    @classmethod
    def load(cls, filename, N):
        fhistograms_from_file = np.loadtxt(filename)
        num_dirs = fhistograms_from_file.shape[-1]
        fhistograms = np.ndarray((N, N, num_dirs))
        fhistograms[np.triu_indices(N)] = fhistograms_from_file
        return cls(fhistograms)


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

    elif matching == 'optimal':
        shape_distance, matching = optimal_shape_matching(A, B, metric)
        spatial_distance = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                mi, mj = matching[i], matching[j]
                if mi <= mj:
                    spatial_distance += hdist.distance(A[i, j], B[mi, mj],
                                                       metric)
                else:
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
