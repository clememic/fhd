"""
FHD descriptors module.
"""

import ctypes
import glob
import heapq
import os

import numpy as np
import pymeanshift as pyms
from skimage.io import ImageCollection, imsave
from skimage.morphology import erosion, square
from sklearn.cluster import KMeans

import datasets
import hdist

# Load C shared library for FHistograms
libfh = os.path.join(os.path.dirname(__file__), 'libfhistograms_raster.so')
clib = ctypes.cdll.LoadLibrary(libfh)

# Different matching strategies
MATCHINGS = ('default', 'greedy', 'optimal')

# Conversion from RGB to Luma value
RGB_TO_LUMA = (0.299, 0.587, 0.114)

EXPERIMENTS_PATH = os.path.join(os.path.dirname(__file__), 'experiments')


def meanshift(image, spatial_radius, range_radius, min_density):
    """
    Segment an image using the meanshift clustering algorithm.

    Parameters
    ----------
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
    n_modes : int
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
    segmented, labels, n_modes = pyms.segment(image, hs, hr, M)
    return segmented, n_modes


def kmeans(image, n_clusters):
    """
    Segment an image using the kmeans clustering algorithm.

    Parameters
    ----------
    image : array_like
        Input image.
    n_clusters : int
        The number of clusters to form.

    Returns
    -------
    segmented : array_like
        Segmented image.
    clusters : array_like, shape(num_clusters, n_features)
        The formed clusters.

    Notes
    -----
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
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(samples)
    clusters = kmeans.cluster_centers_.astype(np.uint8)
    for index in range(samples.shape[0]):
        samples[index] = clusters[labels[index]]
    return segmented, clusters


def decomposition(image, n_layers, spatial_radius, range_radius, min_density,
                  filter_bg=True):
    """
    Decompose an input image into multiple layers.

    Returns the obtained layers as well as the intermediate segmentation steps.

    Parameters
    ----------
    image : array_like
        Input image.
    n_layers : int
        The number of layers to decompose into.
    spatial_radius : int
        Spatial radius of the search window.
    range_radius : float
        Range radius parameter of the search window.
    min_density : int
        Minimum size of a region in the segmented image.
    filter_bg : bool, optional, default: True
        Wheter the background of the image should be filtered.

    Returns
    -------
    layers : list of binary images
        Decomposition of the image into binary layers.
    kmeans_segmented : ndarray
        Intermediate kmeans segmentation.
    meanshift_segmented : ndarray
        Intermediate meanshift segmentation.

    Notes
    -----
    If `filter_bg` is True, the kmeans segmentation step will form one
    more cluster but the one assigned to the top left pixel will be ignored.
    Layers are sorted by decreasing luma.
    Layers are binary eroded by a 3 * 3 structuring element.

    See also
    --------
    meanshift : Meanshift clustering algorithm.
    kmeans : KMeans clustering algorithm.

    """
    ms = meanshift(image, spatial_radius, range_radius, min_density)
    meanshift_segmented = ms[0]

    n_clusters = n_layers
    if filter_bg:
        n_clusters += 1
    kmeans_segmented, clusters = kmeans(meanshift_segmented, n_clusters)
    if filter_bg:
        # Filter the cluster assigned to the top-left pixel
        clusters = clusters[~(clusters == kmeans_segmented[0, 0]).all(-1)]
    # Clusters are sorted by decreasing luma
    clusters = sorted(clusters, key=lambda c: c.dot(RGB_TO_LUMA), reverse=True)

    width, height = image.shape[:2]
    layers = np.zeros((n_layers, width, height), np.uint8)
    for index, cluster in enumerate(clusters):
        mask = np.where((kmeans_segmented == cluster).all(-1))
        layers[index][mask] = 255
        layers[index] = erosion(layers[index], selem=square(3))

    return layers, kmeans_segmented, meanshift_segmented


def fhistogram(a, b=None, n_dirs=180, force_type=0.0):
    """
    Compute an FHistogram between two binary images.

    The FHistogram is computed between `a` and `b` images, along `n_dirs`
    directions and using the attraction force `force_type`.

    Parameters
    ----------
    a, b : (w, h,) array_like
        The two binary images to consider. `a` and `b` must have the same size.
        If `b` is None, the FHistogram is computer between `a` and `a`.
    n_dirs : int
        The number of directions to consider. Default is 180.
    force_type : float
        The attraction force used to compute the FHistogram. Default is 0.0.

    Returns
    -------
    fh : (n_dirs,) ndarray
        The FHistogram between `a` and `b` along `n_dirs` directions using the
        attraction force `force_type`.

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
    fhistogram = np.ndarray(n_dirs)
    height, width = a.shape
    clib.FRHistogram_CrispRaster(
        fhistogram.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n_dirs),
        ctypes.c_double(force_type),
        a.ctypes.data_as(ctypes.c_char_p),
        b.ctypes.data_as(ctypes.c_char_p),
        ctypes.c_int(width),
        ctypes.c_int(height))
    return fhistogram


def fhd(layers, n_dirs=180, shape_force=0.0, spatial_force=0.0):
    """
    Compute an FHD descriptor with layers extracted from a segmented image.

    Arguments
    ---------
    layers : list of binary images
        Binary layers extracted from a segmented image.
    n_dirs : int, optional, default=180
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
    fhistograms = np.ndarray((N, N, n_dirs))
    for i in range(N):
        for j in range(i, N):
            fhistograms[i, j] = fhistogram(
                layers[i], layers[j], n_dirs,
                shape_force if i == j else spatial_force)
    return FHD(fhistograms)


def from_file(filename, N):
    """Load an FHD descriptor from file."""
    fhistograms_from_file = np.loadtxt(filename)
    n_dirs = fhistograms_from_file.shape[-1]
    fhistograms = np.ndarray((N, N, n_dirs))
    fhistograms[np.triu_indices(N)] = fhistograms_from_file
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
    n_dirs : int
        The number of directions for each FHistogram of the FHD.
    fhistograms : (N, N, n_dirs) array_like
        The underlying FHistograms of the FHD.

    """

    def __init__(self, fhistograms):
        """Create an FHD descriptor."""
        self.N = fhistograms.shape[0]
        self.n_dirs = fhistograms.shape[-1]
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

    def normalized(self):
        """Return a normalized copy of the FHD."""
        normalized = FHD(self.fhistograms.copy())
        for i in range(normalized.N):
            for j in range(i, normalized.N):
                normalized.fhistograms[i, j] /= normalized[i, j].max()
        return normalized

    def dump(self, filename):
        """Dump FHD descriptor to file."""
        np.savetxt(filename, self.fhistograms[np.triu_indices(self.N)])


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
    if matching not in MATCHINGS:
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
                    pivot = B.n_dirs // 2
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
                    pivot = B.n_dirs // 2
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


def nearest_neighbors(test_set, train_set, n_neighbors=1, metric='L2',
                      matching='default', alpha=None):
    """
    k-Nearest Neighbors algorithm for FHD descriptors.

    Parameters
    ----------
    test_set : list of FHD descriptors
        The test set.
    train_set : list of FHD descriptors
        The training set.
    n_neighbors : int, optional, default = 1
        The amount of nearest neighbors to search for.

    Returns
    -------
    nearest_neighbors : list
    For each entry of the test set, a list of its nearest neighbors in the
    train set, each in the form of a tuple (distance, index).

    Notes
    -----
    Because the FHD descriptors have a lot of dimensions, nearest neighbors are
    searched using a naive brute-force implementation.

    """
    test_set = np.atleast_1d(test_set)
    nearest_neighbors = []
    for test in test_set:
        heap = []
        for index, train in enumerate(train_set):
            d = distance(test, train, metric, matching, alpha)
            heapq.heappush(heap, (d, index))
        nearest_neighbors += [heapq.nsmallest(n_neighbors, heap)]
    return nearest_neighbors


def run_experiment(dataset, n_layers, n_dirs, shape_force, spatial_force,
                   spatial_radius, range_radius, min_density):
    """
    Run an FHD experiment on a dataset.

    This function loops over the images in a dataset and computes an FHD
    descriptor for each of them, for a giver set of parameters. The descriptor
    as well as the intermediate decomposition steps are stored on the disk.

    Parameters
    ----------
    dataset : str
        The name of the dataset.
    n_layers : int
        Number of layers for the decomposition.
    n_dirs : int
        Number of directions to consider for the FHistograms.
    shape_force : int
        Attraction force used for shape FHistograms.
    spatial_force : int
        Attraction force used for spatial relations FHistograms.
    spatial_radius : int
        Spatial radius of the Meanshift's search window.
    range_radius : float
        Range radius of the Meanshift's search window.
    min_density : int
        Minimum size of a region for the Meanshift.

    """
    dataset = datasets.load(dataset)
    num_images = len(dataset.images)

    # Path of the experiment
    relpath = '{}/{}-{}-{}-{}-{}-{}-{}'.format(dataset.name, n_layers, n_dirs,
        shape_force, spatial_force, spatial_radius, range_radius, min_density)
    path = os.path.join(EXPERIMENTS_PATH, relpath)

    for i, image in enumerate(dataset.images):
        # Print progress
        print('[{}/{}] {}'.format(str(i + 1).zfill(len(str(num_images))),
            num_images, dataset.images.files[i]))

        # Decomposition into layers and FHD computation
        layers, kmeans, meanshift = decomposition(image, n_layers,
            spatial_radius, range_radius, min_density)
        fhd_descriptor = fhd(layers, n_dirs, shape_force, spatial_force)

        # Directory for each image in the dataset
        name = os.path.splitext(dataset.images.files[i].split('/')[-1])[0]
        curr_path = os.path.join(path, name)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)

        # Dump files (FHD, decomposition steps)
        fhd_descriptor.dump(os.path.join(curr_path, 'fhd.txt'))
        for i, layer in enumerate(layers):
            imsave(os.path.join(curr_path, 'layers-{}.png'.format(i)), layer)
        imsave(os.path.join(curr_path, 'kmeans.png'), kmeans)
        imsave(os.path.join(curr_path, 'meanshift.png'), meanshift)


def load_experiment(path):
    """
    Load an FHD experiment located at given path.

    Parameters
    ----------
    path : str
        The path of the experiment to load.

    Returns
    -------
    experiment : Bunch
        The loaded FHD experiment.

    """
    path = os.path.normpath(path)
    dataset = datasets.load(path.split('/')[-2])
    n_layers = int(path.split('/')[-1].split('-')[0])

    fhd_files = sorted(glob.glob(os.path.join(path, '*/fhd.txt')))
    fhds = np.array([from_file(fhd_file, n_layers) for fhd_file in fhd_files])

    experiment = dataset
    experiment['path'] = path
    experiment['n_layers'] = n_layers
    experiment['fhds'] = fhds
    return experiment
