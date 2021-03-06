"""
FHD descriptors module.
"""

import ctypes
import glob
import heapq
import os

import numpy as np
import pymeanshift as pyms
from scipy.stats import mode
from skimage.io import ImageCollection, imsave
from skimage.morphology import erosion, square
from sklearn.cluster import KMeans
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel
from joblib import delayed

import datasets
import hdist

# Load C shared library for FHistograms
libfh = os.path.join(os.path.dirname(__file__), 'libfhistograms_raster.so')
clib = ctypes.cdll.LoadLibrary(libfh)

# Different alignments and matching strategies
ALIGNMENTS = ('luminance', 'size')
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
    fhd = np.zeros((N, N, n_dirs))
    for i in range(N):
        for j in range(i, N):
            fhd[i, j] = fhistogram(
                layers[i], layers[j], n_dirs,
                shape_force if i == j else spatial_force)
    return fhd


def from_file(filename, N, alignment='luminance'):
    """
    Load an FHD descriptor from file.

    Parameters
    ----------
    filename : str
        Path of the FHD file.
    N : int
        Number of layers in the FHD.
    alignment : str, default = 'luminance'
        The alignment of layers for the FHD.

    Returns
    -------
    The FHD descriptor array.

    Notes
    -----
    Size alignment is done by sorting layers according to the (approximate)
    integral of each shape descriptor (FHistograms on the diagonal). This way,
    we don't have to load the underlying binary image of each layer.

    """
    if alignment not in ALIGNMENTS:
        raise ValueError("Incorrect alignment.")

    # Load FHD from file, default alignment of layers should be by luminance
    fhd_from_file = np.loadtxt(filename)
    n_dirs = fhd_from_file.shape[-1]
    fhd = np.zeros((N, N, n_dirs))
    fhd[np.triu_indices(N)] = fhd_from_file

    if alignment == 'size':
        # Reorganize the FHD with layers sorted by decreasing size
        # Just reassign a new FHD (in-place = brainfuck)
        sorted_fhd = np.zeros((N, N, n_dirs))
        argsort_size = np.argsort([np.trapz(fhd[i, i])
                                   for i in range(N)])[::-1]
        for i in range(N):
            si = argsort_size[i]
            sorted_fhd[si, si] = fhd[i, i]
            for j in range(i + 1, N):
                sj = argsort_size[j]
                if si < sj:
                    sorted_fhd[si, sj] = fhd[i, j]
                else:
                    # si > sj (they cannot be equal)
                    # Swap new indices to stay on upper triangle, and shift
                    # the FHistogram (semantic inverse)
                    pivot = fhd.shape[-1] // 2
                    sorted_fhd[sj, si] = np.roll(fhd[i, j], pivot)
        fhd = sorted_fhd

    return fhd

def to_file(filename, fhd):
    """Dump FHD descriptor to file."""
    np.savetxt(filename, fhd[np.triu_indices(fhd.shape[0])])


def distance(A, B, metric='L2', matching='default', alpha=0.5):
    """
    Distance between two FHD descriptors.

    Parameters
    ----------
    A, B : ndarrays
        The FHD descriptors.
    metric : str
        The distance metric used to compare histograms.
    matching : str
        The matching strategy used.
    alpha : float between 0 and 1, default is 0.5
        Weight given to the distance between shapes compared to the distance
        between spatial relations.

    Returns
    -------
    The distance between the two FHD descriptors.

    """
    if A.shape != B.shape:
        raise ValueError('A and B should have the same shape.')
    N = A.shape[0]
    if not 0 <= alpha <= 1:
        raise ValueError('alpha should be between 0 and 1.')
    if matching not in MATCHINGS:
        raise ValueError('Incorrect matching strategy.')

    if matching == 'default':
        matching = [i for i in range(N)]

    elif matching == 'greedy':
        # Two-way greedy matchings, keep the minimum sum of pairwise distances
        matching_AB, pairwise_distances_AB = greedy_matching(A, B, metric)
        matching_BA, pairwise_distances_BA = greedy_matching(B, A, metric)
        if pairwise_distances_AB.sum() <= pairwise_distances_BA.sum():
            matching = matching_AB
        else:
            matching = matching_BA

    elif matching == 'optimal':
        matching, _ = optimal_matching(A, B, metric)

    shape_distance = 0.0
    spatial_distance = 0.0
    for i in range(N):
        shape_distance += hdist.distance(A[i, i], B[i, i], metric)
        for j in range(i + 1, N):
            mi, mj = matching[i], matching[j]
            if mi <= mj:
                spatial_distance += hdist.distance(
                    A[i, j], B[mi, mj], metric)
            else:
                # In this case, the FHistogram in B must be shifted by
                # half its size to mimic the lower diagonal of the matrix
                mi, mj = mj, mi
                pivot = B.shape[-1] // 2
                spatial_distance += hdist.distance(
                    A[i, j], np.roll(B[mi, mj], pivot), metric)

    # Same weight to each histogram
    shape_distance /= N
    spatial_distance /= ((N * (N - 1)) / 2)

    return (alpha * shape_distance) + ((1 - alpha) * spatial_distance)


def greedy_matching(A, B, metric='L2'):
    """
    Return a greedy matching between the layers of two FHD descriptors.

    The matching is based on shape FHistograms (diagonal of the FHD) and is
    computed in a greedy way. That is, each shape of A is matched with its
    closest shape in B, according to a distance metric.

    Parameters
    ----------
    A, B : ndarrays
        The two FHD descriptors.
    metric : str
        The distance metric used to compare histograms.

    Returns
    ------
    matching : list
        For each layer of `A`, its matched layer in `B`.
    pairwise_distances : ndarray
        Distances between the shape FHistograms of the matched layers.

    Notes
    -----
    The function is not symmetric. That is, the greedy matching from `A` to `B`
    can be different than the one from `B` to `A`. Symmetry can be achieved by
    computing the two greedy matchings and keeping the minimum resulting sum of
    pairwise distances.

    """
    if A.shape != B.shape:
        raise ValueError('A and B should have the same shape.')
    N = A.shape[0]

    matching = []
    pairwise_distances = np.ndarray(N)
    available_matchings = [i for i in range(N)]

    for i in range(N):
        distances = [(hdist.distance(A[i, i], B[j, j], metric,
                                     normalized=False), j)
                     for j in available_matchings]
        min_distance, min_matching = min(distances)
        available_matchings.remove(min_matching)
        matching.append(min_matching)
        pairwise_distances[i] = min_distance

    return matching, pairwise_distances


def optimal_matching(A, B, metric='L2'):
    """
    Return an optimal matching between the layers of  two FHD descriptors.

    The matching is based on shape FHistograms (diagonal of the FHD). This
    function computes the sum of pairwise distances between all possible
    permutations of shape FHistograms of `A` and `B` and returns the minimum
    one.

    Parameters
    ----------
    A, B : ndarrays
        The two FHD descriptors
    metric : str, optional
        The distance metric used to compare histograms.

    Returns
    -------
    matching : dict
        For each layer of `A`, its matched layer in `B`.
    distance : float
        Sum of pairwise distances between the shapes of the matched layers.

    Notes
    -----
    Careful, complexity is N! where N is the number of layers.

    """
    if A.shape != B.shape:
        raise ValueError('A and B should have the same shape.')
    N = A.shape[0]

    # Distances between all pairwise shapes of A and B (N * N matrix)
    distances = np.ndarray((N, N))
    for i in range(N):
        for j in range(N):
            distances[i, j] = hdist.distance(A[i, i], B[j, j], metric)

    # Sum of pairwise distances for all possible permutations
    matchings = {}
    from itertools import permutations
    for permutation in permutations(range(N)):
        distance = 0.0
        for i, j in enumerate(permutation):
            distance += distances[i, j]
        matchings[distance] = permutation

    # Return the best matching
    distance = min(matchings.keys())
    best_permutation = matchings[distance]
    matching = []
    for i in range(N):
        matching.append(best_permutation[i])
    return matching, distance


def nearest_neighbors(test_sample, train_set, n_neighbors=1, metric='L2',
                      matching='default', alpha=0.5):
    """
    k-Nearest Neighbors algorithm for FHD descriptors.

    Parameters
    ----------
    test_sample : FHD descriptor
        The test sample.
    train_set : list of FHD descriptors
        The training set.
    n_neighbors : int, optional, default = 1
        The amount of nearest neighbors to search for.

    Returns
    -------
    nearest_neighbors : list
    The list of nearest neighbors in the train set for the test sample, each in
    the form of a tuple (distance, index).

    Notes
    -----
    Because the FHD descriptors have a lot of dimensions, nearest neighbors are
    searched using a naive brute-force implementation.

    """
    nearest_neighbors = []
    heap = []
    for index, train in enumerate(train_set):
        d = distance(test_sample, train, metric, matching, alpha)
        heapq.heappush(heap, (d, index))
    nearest_neighbors += heapq.nsmallest(n_neighbors, heap)
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
        to_file(os.path.join(curr_path, 'fhd.txt'), fhd_descriptor)
        for i, layer in enumerate(layers):
            imsave(os.path.join(curr_path, 'layers-{}.png'.format(i)), layer)
        imsave(os.path.join(curr_path, 'kmeans.png'), kmeans)
        imsave(os.path.join(curr_path, 'meanshift.png'), meanshift)


def load_experiment(path, alignment='luminance'):
    """
    Load an FHD experiment located at given path.

    Parameters
    ----------
    path : str
        The path of the experiment to load.
    alignment : str, default = 'luminance'
        Default alignment of layers for the FHDs.

    Returns
    -------
    experiment : Bunch
        The loaded FHD experiment.

    Notes
    -----
    FHistograms are scaled between [0, 1] globally (no loss of information) and
    independently for shapes and spatial relations.

    """
    path = os.path.normpath(path)
    dataset = datasets.load(path.split('/')[-2])
    n_layers = int(path.split('/')[-1].split('-')[0])

    if alignment not in ALIGNMENTS:
        raise ValueError("Incorrect alignment.")

    fhd_files = sorted(glob.glob(os.path.join(path, '*/fhd.txt')))
    fhds = np.array([from_file(fhd_file, n_layers, alignment=alignment)
                     for fhd_file in fhd_files])

    # Feature scaling (shapes and spatial relations independently)
    shapes = np.vstack([_[np.diag_indices(n_layers)] for _ in fhds])
    spatials = np.vstack([_[np.triu_indices(n_layers, 1)] for _ in fhds])
    for fhd in fhds:
        fhd[np.diag_indices(n_layers)] -= shapes.min()
        fhd[np.diag_indices(n_layers)] /= (shapes.max() - shapes.min())
        fhd[np.triu_indices(n_layers, 1)] -= spatials.min()
        fhd[np.triu_indices(n_layers, 1)] /= (spatials.max() - spatials.min())

    experiment = dataset
    experiment['path'] = path
    experiment['n_layers'] = n_layers
    experiment['fhds'] = fhds
    return experiment


def cross_validate(experiment, n_neighbors=1, metric='L2', matching='default',
                   alpha=0.5):
    """
    Leave-one-out cross validation for an experiment.

    A `predictions` entry will be added in the experiment and can be compared
    with ground truth labels.

    Parameters
    ----------
    experiment : Bunch
        An FHD experiment on a labeled dataset.
    n_neighbors : int, optional, default = 1
        The number of nearest neighbors to consider.
    metric : str
        The distance metric used to compare histograms.
    matching : str
        The matching strategy used.
    alpha : float between 0 and 1, default is 0.5
        Weight given to the distance between shapes compared to the distance
        between spatial relations.

    Notes
    -----
    When `n_neighbors` > 1, a simple majority vote is used to choose the
    predicted label.
    Cross validation is parallelized across all CPUs with joblib.

    """
    n_samples = len(experiment.labels)
    loo = LeaveOneOut(n_samples)
    experiment['predictions'] = np.array(Parallel(n_jobs=-1)(
        delayed(_cross_validate_single)(experiment, n_neighbors, metric,
                                        matching, alpha, test[0], train)
                for train, test in loo))


def _cross_validate_single(experiment, n_neighbors, metric, matching, alpha,
                           test, train):
    nn = nearest_neighbors(experiment.fhds[test], experiment.fhds[train],
                           n_neighbors, metric, matching, alpha)
    labels = [experiment.labels[train][n[1]] for n in nn]
    prediction = int(mode(labels)[0][0])
    return prediction


def alpha_learning(experiment, n_neighbors=1, metric='L2', matching='default',
                   learning_type='by_class'):
    """
    Learn 'optimal' alpha weights on an experiment.

    Parameters
    ----------
    experiment : Bunch
        An FHD experiment on a labeled dataset.
    n_neighbors : int, optional, default = 1
        The number of nearest neighbors to consider.
    metric : str
        The distance metric used to compare histograms.
    matching : str
        The matching strategy used.
    learning_type : str
        The learning type. Possible values are 'by_class' and 'global'.
        'by_class' computes optimal alpha values for each class of the dataset,
         whereas 'global' computes a global optimal alpha value for the dataset.

    Returns
    -------
    learned : list or float
        For 'by_class', a list of learned alpha values. For 'global', the
        learned global alpha value.

    Notes
    -----
    alpha values between 0.1 and 0.9 are tried with a step of 0.1.
    The default alpha value is 0.5 (for example if a class is not recognized).

    """
    if learning_type not in ['by_class', 'global']:
        raise ValueError('Wrong learning type.')
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # alphas = np.linspace(0, 1)
    n_samples = len(experiment.labels)
    n_labels = len(np.unique(experiment.labels))

    if learning_type =='by_class':
        # Learn best alpha values for each class
        best_alphas = np.zeros(n_labels) + 0.5
        best_rec_rates = np.zeros(n_labels)
        for alpha in alphas:
            cross_validate(experiment, n_neighbors=n_neighbors, metric=metric,
                           matching=matching, alpha=alpha)
            # Compute recognition rate for each class
            true_positives = np.zeros(n_labels)
            for i in range(n_samples):
                if experiment.predictions[i] == experiment.labels[i]:
                    true_positives[experiment.labels[i] - 1] += 1
            rec_rates = true_positives / np.bincount(experiment.labels)[1:]
            # Keep track of best alpha values for each class
            for i, rec_rate in enumerate(rec_rates):
                if rec_rate > best_rec_rates[i]:
                    best_rec_rates[i] = rec_rate
                    best_alphas[i] = alpha
        return best_alphas

    elif learning_type=='global':
        # Learn best global alpha value
        best_alpha = 0.5
        best_rec_rate = 0.0
        for alpha in alphas:
            cross_validate(experiment, n_neighbors=n_neighbors, metric=metric,
                           matching=matching, alpha=alpha)
            # Just keep track of global recognition rate for each alpha
            rec_rate = accuracy_score(experiment.labels, experiment.predictions)
            if rec_rate > best_rec_rate:
                best_rec_rate = rec_rate
                best_alpha = alpha
        return best_alpha


def dump_results(experiment):
    """
    Dump results of an experiment.

    Parameters
    ----------
    experiment : Bunch
        An FHD experiment on a labeled dataset.

    """
    path = experiment.path.replace('experiments', 'results')
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, '{}-{}.txt'.format(metric, matching))
    results_file = open(path, 'w+')
    accuracy = accuracy_score(experiment.labels, experiment.predictions)
    results_file.write('Recognition rate: ' + str(np.round(accuracy, 3) * 100))
    results_file.write('\n')
    results_file.write(classification_report(experiment.labels,
                                             experiment.predictions))
    results_file.close()
